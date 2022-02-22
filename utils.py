import os
import sys
import tifffile
import numpy as np
from PIL import Image
from pycocotools import coco
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def stack_tifs(dir):
    """Stacks .tif files in a single stack

    Args:
        dir (string): input directory path

    Returns:
        A .tif stack in the ~/variables/ subfolder with 
            (pages, height, width) as dimensions
        A .txt file with the original .tif filenames
            in the ~/variables/ subfolder
    """

    tif_set = os.listdir(dir)
    tif_set = sorted(tif_set)
    out_folder = f'{dir}variables/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    with tifffile.TiffWriter(f'{out_folder}stack.tif') as tif_stack:
        with open(f'{out_folder}filenames.txt', 'w') as tif_filename_file:
            for filename in tif_set:
                if filename.endswith('.tif'):
                    tif_stack.save(
                        tifffile.imread(f'{dir}{filename}'),
                        photometric='minisblack',
                    )
                    file_line = f'{filename}\n'
                    tif_filename_file.write(file_line)


def tif_stack2arr(dir):
    """Transforms .tif stack into numpy array

    Args:
        dir (string): input directory path

    Returns:
        A 3D numpy array .npy file in the ~/variables/ subfolder
            with (pages, height, width) as dimensions
    """

    out_folder = f'{dir}variables/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    
    images = []
    tif_stack = Image.open(f'{out_folder}stack.tif')
    for page in range(tif_stack.n_frames):
        tif_stack.seek(page)
        images.append(np.array(tif_stack))
    array = np.array(images)
    np.save(f'{out_folder}array.npy', array)


def json2array(filepath, create_tifs=False):
    """Transforms a .json segmentation file into a binary mask array
    
    Note:
        Overlapping polygons are segmented with a pixel-wide boundary
        All outputs are put in a folder named after the input file

    Args:
        filepath (string): input file path
        create_tifs (boolean): binary masks are also to be saved 
            as individual .tif files
    
    Returns:
        A [0, 255] 3D numpy array .npy file in the ~/variables/ 
            subfolder with (..., height, width) as dimensions
        A .txt file with the original .tif filenames in the 
            ~/variables/ subfolder
        (Optional) Binary [0, 255] segmentation mask .tif files 
    """

    json_file = coco.COCO(filepath)
    imgIds = json_file.getImgIds()
    Imgs = json_file.loadImgs(imgIds)
    nImages = len(imgIds)
    height = Imgs[0]['height']
    width = Imgs[0]['width']
    
    out_folder_tifs = f'{filepath.rsplit(".", 1)[0]}/' 
    if not os.path.isdir(out_folder_tifs):
        os.makedirs(out_folder_tifs)
    out_folder_variables = f'{out_folder_tifs}variables/' 
    if not os.path.isdir(out_folder_variables):
        os.makedirs(out_folder_variables)

    cell_brightness = 0
    mask_array = np.zeros([nImages, height, width], dtype = 'uint8') 
    with open(f'{out_folder_variables}tif_filenames.txt', 'w') as tif_filenames:
        for iImage in range(len(imgIds)):
            annIds_image = json_file.getAnnIds(imgIds = imgIds[iImage])
            anns_image = json_file.loadAnns(annIds_image)
            orig_filename = Imgs[iImage]['file_name']
            out_file_line = f'{orig_filename}\n'
            tif_filenames.write(out_file_line)
            
            for iAnns in range(len(anns_image)):
                cell_brightness += 1; 
                if cell_brightness > 255: cell_brightness = 1
                annot_mask = json_file.annToMask(anns_image[iAnns]) * cell_brightness
                mask_array[iImage,:,:] = np.maximum(mask_array[iImage,:,:], annot_mask)

            for iY in range(height):
                for iX in range(width):
                    current = mask_array[iImage,iY,iX]
                    bottom = mask_array[iImage,min(iY+1,height-1),iX]
                    bottom_right = mask_array[iImage,min(iY+1,height-1),min(iX+1,width-1)]
                    right = mask_array[iImage,iY,min(iX+1,width-1)]
                    top_right = mask_array[iImage,max(0,iY-1),min(iX+1,width-1)]
                    if not {bottom, bottom_right, right, top_right}.issubset({0, current}):
                        mask_array[iImage,iY,iX] = 0
            
            mask_array[iImage,:,:] = 255 * np.uint8(mask_array[iImage,:,:] > 0)

            if create_tifs:
                with tifffile.TiffWriter(f'{out_folder_tifs}{orig_filename}') as tif:
                    tif.write(
                        mask_array[iImage,:,:],
                        photometric='minisblack',
                    ) 
    np.save(f'{out_folder_variables}array.npy', mask_array)


def stack_orient(dir, square=False):
    """Returns a concatenated stack of all 2D input flips and rotations
        of a 3D numpy array with (..., height, width) as dimensions
        found in the input directory subfolder ~/variables/
    
    Note:
        The order of orientations in the first array dimension is:
            - Original
            - Vertical flip / Horizontal mirror
            - Horizontal flip / Vertical mirror
            - Vertical + Horizontal flip
            The last four stacks equal the first four,
            but rotated 90 degrees counterclockwise
        The array will be filled out with zeros if not square so
            rotations can be stacked directly on the original
    
    Args:
        dir (string): the input directory path
        square (boolean): the input array is square-shaped
    
    Returns:
        A 3D numpy array .npy file with (..., height, width) 
            as dimensions
        A .txt file with unique, appropriate .tif filenames
    """

    out_folder = f'{dir}variables/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    arr = np.load(f'{out_folder}array.npy')
    filenames = []
    with open(f'{out_folder}filenames.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            filenames.append(args[0])

    appendices = ['_orig', 
                  '_vflip', 
                  '_hflip', 
                  '_vhflip', 
                  '_orig_90rot', 
                  '_vflip_90rot', 
                  '_hflip_90rot', 
                  '_vhflip_90rot'] 
    stacked_filenames = [filename.rsplit('.', 1)[0] 
                         for appendix in appendices 
                         for filename in filenames]
    stacked_appendices = [appendix 
                          for appendix in appendices 
                          for filename in filenames]
    oriented_filenames = ["{}{}.tif".format(stacked_filename, stacked_appendix) 
                          for stacked_filename, stacked_appendix in zip(stacked_filenames, stacked_appendices)]

    arr_stack = np.concatenate((arr,
                            np.flip(arr, 1),
                            np.flip(arr, 2),
                            np.flip(np.flip(arr, 2),1)))

    if ~square:
        max_img_length = max(arr_stack.shape[1], arr_stack.shape[2])
        square_arr_stack_shape = (arr_stack.shape[0], max_img_length, max_img_length)
        square_arr_stack = np.zeros(square_arr_stack_shape, dtype='uint8')
        square_arr_stack[:arr_stack.shape[0], :arr_stack.shape[1], :arr_stack.shape[2]] = arr_stack
        oriented_array = np.concatenate((square_arr_stack,
                                         np.rot90(square_arr_stack, axes=(1, 2))
                                        ))
        np.save(f'{out_folder}oriented_array.npy', oriented_array)
    else:
        oriented_array = np.concatenate((arr_stack,
                                         np.rot90(arr_stack, axes=(1, 2))
                                        ))
        np.save(f'{out_folder}oriented_array.npy', oriented_array)

    with open(f'{out_folder}oriented_filenames.txt', 'w') as oriented_filename_file:
        for oriented_filename in oriented_filenames:
            file_line = f'{oriented_filename}\n'
            oriented_filename_file.write(file_line)

    # plt.imshow(oriented_array[800,:,:], cmap='gray', vmin=0, vmax=255)
    # plt.show()