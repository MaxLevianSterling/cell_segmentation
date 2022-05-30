from curses.panel import top_panel
import os
import sys
from re import L
import tifffile
import numpy        as np
from PIL            import Image
from pycocotools    import coco
from utils          import HiddenPrints
from utils          import path_gen
from eval           import binary2json
from tifffile       import imwrite
import copy


def stack_tifs(dir):
    """Stacks .tif files in a single stack

    Note:
        Image data must be grayscale
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            <data_subset>/<.tif files>
                        annotations/
                            <data_subset>/<.tif files>
    
    Args:
        dir (string): input directory path

    Returns:
        A .tif stack in the ~/variables/ subfolder with 
            (pages, height, width) as dimensions
        A .txt file with the original .tif filenames
            in the ~/variables/ subfolder
    """

    # Get directory files and sort them
    file_set = os.listdir(dir)
    file_set = sorted(file_set)

    # Make output folder if necessary
    out_folder = f'{dir}variables/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Write file names and stack .tif files
    with tifffile.TiffWriter(f'{out_folder}stack.tif') as tif_stack:
        with open(f'{out_folder}filenames.txt', 'w') as tif_filename_file:
            for filename in file_set:
                if filename.endswith('.tif'):
                    tif_stack.save(
                        tifffile.imread(f'{dir}{filename}'),
                        photometric='minisblack',
                    )
                    file_line = f'{filename}\n'
                    tif_filename_file.write(file_line)


def tif_stack2array(dir):
    """Transforms .tif stack into numpy array
    
    Note:
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            <data_subset>/<.tif files>
                        annotations/
                            <data_subset>/<.tif files>
    
    Args:
        dir (string): input directory path

    Returns:
        A 3D numpy array .npy file in the ~/variables/ subfolder
            with (pages, height, width) as dimensions
    """
    
    # Initialize internal variables
    images = []
    out_folder = f'{dir}variables/'

    # Open .tif stack
    tif_stack = Image.open(f'{out_folder}stack.tif')
    
    # Get .tif pages into list
    for page in range(tif_stack.n_frames):
        tif_stack.seek(page)
        images.append(np.array(tif_stack))
    
    # Convert list into numpy array
    array = np.array(images)

    # Save numpy array
    np.save(f'{out_folder}array.npy', array)


def json2array(filepath, annot_type, create_tifs=False):
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

    # Read COCO .json file
    json_file = coco.COCO(filepath)

    # Load image key values
    imgIds = json_file.getImgIds()
    Imgs = json_file.loadImgs(imgIds)

    # Create output folders if necessary
    out_folder_tifs_soma = f'{filepath.rsplit(".", 1)[0]}/soma/' 
    if not os.path.isdir(out_folder_tifs_soma):
        os.makedirs(out_folder_tifs_soma)
    out_folder_tifs_mem = f'{filepath.rsplit(".", 1)[0]}/mem/' 
    if not os.path.isdir(out_folder_tifs_mem):
        os.makedirs(out_folder_tifs_mem)
    out_folder_tifs_thick_mem = f'{filepath.rsplit(".", 1)[0]}/thick_mem/' 
    if not os.path.isdir(out_folder_tifs_thick_mem):
        os.makedirs(out_folder_tifs_thick_mem)
    out_folder_variables = f'{filepath.rsplit(".", 1)[0]}/variables/' 
    if not os.path.isdir(out_folder_variables):
        os.makedirs(out_folder_variables)

    # Initialize internal variables
    n_imgs = len(imgIds)
    h = Imgs[0]['height']
    w = Imgs[0]['width'] 
    brightness = 0
    mask_array = np.zeros([n_imgs, h, w], dtype = 'uint8') 

    # Make binary masks, save as numpy array and optionally .tif files 
    with open(f'{out_folder_variables}tif_filenames.txt', 'w') \
            as tif_filenames:
        for iI in range(n_imgs):

            # Save filename identifiers
            orig_filename = Imgs[iI]['file_name']
            out_file_line = f'{orig_filename},{imgIds[iI]}\n'
            tif_filenames.write(out_file_line)
            
            # Get all annotation key values
            annIds_image = json_file.getAnnIds(imgIds = imgIds[iI])
            anns_image = json_file.loadAnns(annIds_image)

            # Convert annotation polygons into color array
            for iA in range(len(anns_image)):

                # Keep changing the annotation color
                brightness += 1; 
                if brightness > 254: brightness = 1

                # Get the annotation into the array
                annot_mask = json_file.annToMask(anns_image[iA]) * brightness
                #annot_mask = np.rot90(annot_mask)
                mask_array[iI,:,:] = np.maximum(mask_array[iI,:,:], annot_mask)

            # Copy mask array
            mask_array_edge = np.copy(mask_array[iI,:,:])

            # Create a neat one-pixel boundary across every color
            for iY in range(h):
                for iX in range(w):
                    current = mask_array[iI, iY,            iX]
                    bot     = mask_array[iI, min(iY+1,h-1), iX]
                    bot_r   = mask_array[iI, min(iY+1,h-1), min(iX+1,w-1)]
                    r       = mask_array[iI, iY,            min(iX+1,w-1)]
                    top_r   = mask_array[iI, max(0,iY-1),   min(iX+1,w-1)]
                    if not {bot, bot_r, r, top_r}.issubset({0, current}):
                        mask_array[iI,iY,iX] = 0
        
            # Convert all annotations to white now that they are separated
            mask_array[iI,:,:] = 255 * np.uint8(mask_array[iI,:,:] > 0)
            
            # Create .tif files if necessary
            if create_tifs:
                imwrite(
                    f'{out_folder_tifs_soma}{orig_filename}',
                    mask_array[iI,:,:],
                    photometric='minisblack',
                ) 

            # # Mark cell membranes only
            # for iY in range(h):
            #     for iX in range(w):
            #         current = mask_array_edge[iY,            iX]
            #         bot_l   = mask_array_edge[min(iY+1,h-1), max(0,iX-1)]
            #         bot     = mask_array_edge[min(iY+1,h-1), iX]
            #         bot_r   = mask_array_edge[min(iY+1,h-1), min(iX+1,w-1)]
            #         r       = mask_array_edge[iY,            min(iX+1,w-1)]
            #         top_r   = mask_array_edge[max(0,iY-1),   min(iX+1,w-1)]
            #         top     = mask_array_edge[max(0,iY-1),   iX]
            #         top_l   = mask_array_edge[max(0,iY-1),   max(0,iX-1)]
            #         l       = mask_array_edge[iY,            max(0,iX-1)]
            #         not_current = set(range(0, current)).union(set(range(current+1, 255)))
            #         if current > 0 and current < 255 and not_current.intersection({bot_l, bot, bot_r, r, top_r, top, top_l, l}):
            #             mask_array_edge[iY,iX] = 255 
        
            # # Convert 
            # mask_array_edge = np.where(mask_array_edge < 255, 0, 255).astype('uint8')                         
        
            # # Create .tif files if necessary
            # if create_tifs:
            #     imwrite(
            #         f'{out_folder_tifs_mem}{orig_filename}',
            #         mask_array_edge,
            #         photometric='minisblack',
            #     ) 

            # # Copy mask array
            # mask_array_thick_edge = np.copy(mask_array[iI,:,:])

            # # Mark cell membranes thickly
            # for iY in range(h):
            #     for iX in range(w):
            #         current = mask_array_thick_edge[iY,            iX]
            #         bot_l   = mask_array_thick_edge[min(iY+1,h-1), max(0,iX-1)]
            #         bot     = mask_array_thick_edge[min(iY+1,h-1), iX]
            #         bot_r   = mask_array_thick_edge[min(iY+1,h-1), min(iX+1,w-1)]
            #         r       = mask_array_thick_edge[iY,            min(iX+1,w-1)]
            #         top_r   = mask_array_thick_edge[max(0,iY-1),   min(iX+1,w-1)]
            #         top     = mask_array_thick_edge[max(0,iY-1),   iX]
            #         top_l   = mask_array_thick_edge[max(0,iY-1),   max(0,iX-1)]
            #         l       = mask_array_thick_edge[iY,            max(0,iX-1)]
            #         if current == 0 and {255, 200}.intersection({bot_l, bot, bot_r, r, top_r, top, top_l, l}):
            #             mask_array_thick_edge[iY,iX] = 50 
            #         if current == 255 and {0, 50}.intersection({bot_l, bot, bot_r, r, top_r, top, top_l, l}):
            #             mask_array_thick_edge[iY,iX] = 200 
        
            # # Convert
            # mask_array_thick_edge = np.where(np.logical_or(mask_array_thick_edge == 50, mask_array_thick_edge == 200), 255, 0).astype('uint8')
            
            # # Create .tif files if necessary
            # if create_tifs:
            #     imwrite(
            #         f'{out_folder_tifs_thick_mem}{orig_filename}',
            #         mask_array_thick_edge,
            #         photometric='minisblack',
            #     )    

            # Display progress
            image_ratio = (iI) / (len(imgIds) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tImages: [{:<{}}] {:.0f}%    ".format(
                    "=" * int(20*image_ratio), 20, 100*image_ratio,
                )
            )
            sys.stdout.flush()
        
    # Save the numpy arrays
    np.save(f'{out_folder_variables}soma_array.npy', mask_array)


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

    # Load filenames
    out_folder = f'{dir}variables/'
    filenames = []
    with open(f'{out_folder}filenames.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            filenames.append(args[0])

    # Load array
    arr = np.load(f'{out_folder}array.npy')

    # Create filename suffices
    suffices = [
        '_orig', 
        '_vflip', 
        '_hflip', 
        '_vhflip', 
        '_orig_90rot', 
        '_vflip_90rot', 
        '_hflip_90rot', 
        '_vhflip_90rot'
    ] 
    
    # Create a shaped stack of filenames
    s_filenames = [
        filename.rsplit('.', 1)[0] 
        for suffix in suffices 
        for filename in filenames
    ]

    # Create a shaped stack of filename suffices
    s_suffices = [
        suffix 
        for suffix in suffices 
        for filename in filenames
    ]

    # Attach both shaped filename stacks together
    oriented_filenames = [
        f'{s_filename}{s_suffix}.tif'
        for s_filename, s_suffix in zip(s_filenames, s_suffices)
    ]

    # Stack flipped array pages
    arr_stack = np.concatenate((
        arr,
        np.flip(arr, 1),
        np.flip(arr, 2),
        np.flip(np.flip(arr, 2),1)
    ))

    # Stack rotated stacks when square
    if ~square:

        # Make stack square by filling out with zeros
        max_img_length = max(arr_stack.shape[1], arr_stack.shape[2])
        square_arr_stack_shape = (
            arr_stack.shape[0], 
            max_img_length, 
            max_img_length
        )
        square_arr_stack = np.zeros(square_arr_stack_shape, dtype='uint8')
        square_arr_stack[
            :arr_stack.shape[0], 
            :arr_stack.shape[1], 
            :arr_stack.shape[2]
        ] = arr_stack
        
        # Stack rotated stacks
        oriented_array = np.concatenate((
            square_arr_stack,
            np.rot90(square_arr_stack, axes=(1, 2))
        ))

        # Save the oriented array
        np.save(f'{out_folder}oriented_array.npy', oriented_array)
    
    else:

        # Stack rotated stacks
        oriented_array = np.concatenate((
            arr_stack,
            np.rot90(arr_stack, axes=(1, 2))
        ))
        
        # Save the oriented array
        np.save(f'{out_folder}oriented_array.npy', oriented_array)

    # Save the new filenames
    with open(f'{out_folder}oriented_filenames.txt', 'w') \
        as oriented_filename_file:
        for oriented_filename in oriented_filenames:
            file_line = f'{oriented_filename}\n'
            oriented_filename_file.write(file_line)


def preprocess(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_type = 'part_set',
    data_subset ='5',
    subset_type = 'test',
    annot_type = 'soma',
    process_mode = 'all',
    orient = False,
):
    """Preprocesses LIVECell image and annotation data

    Note:
        Required file structure:
            <path>/
                data/
                    images/
                        <data_subset>/<.tif files>
                    annotations/
                        <data_subset>/<.tif files>
                        (<.json file>)

    Args:
        path (string): input directory path
        data_set (string): data set 
        data_subset (string): data subset
        orient (bool): whether the data needs to be oriented 
            offline
        
    Returns:
        If the appropriate data is available, after running 
            this script in its entirety, both the image and
            annotation subset folder will contain .tif files; 
            a ~/variables/ subfolder with a .tif stack; an array
            of that stack; a .txt file with all original .tif 
            filenames; a concatenated array with all 8 orientations
            of the original array (orthogonal mirrors + rotations);
            and the .txt file with filenames modified after those
            orientations
    """

    print(f'\nPreprocessing "{data_subset}" data in folder "{path}"')

    ########################### Image data ############################
    
    if process_mode == 'image' or process_mode == 'all':
    
        print('\nChecking for image data... ')

        # Generate folder path string
        img_folder = path_gen([
            path,
            'data',
            data_set,
            data_type,
            data_subset,
            'images',
            subset_type,
        ])

        # Check if image folder exists and has images
        img_folder_exists = os.path.isdir(img_folder)
        n_tifs_imgs = 0
        if img_folder_exists:
            for file in os.listdir(img_folder):
                if file.endswith('.tif'):
                    n_tifs_imgs += 1
        img_folder_has_tifs = n_tifs_imgs > 0
        img_data_exists = img_folder_exists and img_folder_has_tifs
        if img_data_exists:
            print(f'{n_tifs_imgs} .tif files detected in data folder\n')
        else:
            raise RuntimeError('Image data not found.\n')

        # Check if image folder has subfolder with preprocessed variables
        print('Checking for preprocessed image variables... ')
        img_var_folder_exists = os.path.isdir(
            f'{img_folder}variables/'
        )
        img_var_fnames_exists = os.path.isfile(
            f'{img_folder}variables/filenames.txt'
        )
        img_var_tifstack_exists = os.path.isfile(
            f'{img_folder}variables/stack.tif'
        )
        img_var_arr_exists = os.path.isfile(
            f'{img_folder}variables/array.npy'
        )
        img_var_or_arr_exists = os.path.isfile(
            f'{img_folder}variables/oriented_array.npy'
        )
        img_var_or_fnames_exists = os.path.isfile(
            f'{img_folder}variables/oriented_filenames.txt'
        )

        # Display whether all variables are missing
        if not img_var_folder_exists:
            print('No preprocessed variables found...')

        # Display which variables are missing and create them
        if img_var_fnames_exists and \
                img_var_tifstack_exists and \
                img_var_arr_exists and \
                img_var_or_arr_exists and \
                img_var_or_fnames_exists:
            print('All preprocessed variables found...')
        else:
            if not img_var_fnames_exists or not img_var_tifstack_exists:
                print('\tMissing .tif stack or filenames. Creating now...')
                stack_tifs(img_folder)

            if not img_var_arr_exists:
                print('\tMissing array. Creating now...')
                tif_stack2array(img_folder)

            if orient:
                if not img_var_or_arr_exists or not img_var_or_fnames_exists:
                    print(
                        '\tMissing oriented array or oriented filenames.'\
                        'Creating now...'
                    )
                    stack_orient(img_folder)

    ######################## Annotation data #########################

    if process_mode == 'annot' or process_mode == 'all':

        # Generate folder path string
        ann_folder = path_gen([
            path,
            'data',
            data_set,
            data_type,
            data_subset,
            'annotations',
            subset_type,
            annot_type,
        ])
        json_folder = path_gen([
            path,
            'data',
            data_set,
            data_type,
            data_subset,
            'annotations',
        ])

        # Check if annotation folder exists and has images
        ann_folder_exists = os.path.isdir(ann_folder)
        n_tifs_anns = 0
        if ann_folder_exists:
            for file in os.listdir(ann_folder):
                if file.endswith('.tif'):
                    n_tifs_anns += 1

        # Check if annotation folder has subfolder with preprocessed variables
        print('\nChecking for preprocessed annotation variables... ')
        ann_var_folder_exists = os.path.isdir(
            f'{ann_folder}variables/'
        )
        ann_var_fnames_exists = os.path.isfile(
            f'{ann_folder}variables/filenames.txt'
        )
        ann_var_tifstack_exists = os.path.isfile(
            f'{ann_folder}variables/stack.tif'
        )
        ann_var_arr_exists = os.path.isfile(
            f'{ann_folder}variables/array.npy'
        )
        ann_var_or_arr_exists = os.path.isfile(
            f'{ann_folder}variables/oriented_array.npy'
        )
        ann_var_or_fnames_exists = os.path.isfile(
            f'{ann_folder}variables/oriented_filenames.txt'
        )
        ann_var_json_exists = os.path.isfile(
            f'{ann_folder}variables/json.json'
        )

        # Display whether all variables are missing
        if not ann_var_folder_exists:
            print('No preprocessed variables found...')

        # Display which variables are missing and create them
        if ann_var_fnames_exists \
                and ann_var_tifstack_exists \
                and ann_var_arr_exists \
                and ann_var_or_arr_exists \
                and ann_var_or_fnames_exists:
            print('All preprocessed variables found...')
        else:
            print('\nChecking for annotation data...')
            if not ann_folder_exists \
                    or not ann_var_folder_exists \
                    and n_tifs_anns == 0:
                print(
                    '\tMissing .tif files.'\
                    'Checking for .json file... ', end=''
                )
                json_file_exists = os.path.isfile(
                    f'{json_folder}{subset_type}.json'
                )
                if json_file_exists:
                    print(
                        f'Found. Processing...'
                    )      
                    json2array(
                        f'{json_folder}{subset_type}.json', 
                        annot_type=annot_type,
                        create_tifs=True
                    )
                else:
                    raise RuntimeError('Annotation data not found.\n')

            if not ann_var_fnames_exists or not ann_var_tifstack_exists:
                print('\tMissing .tif stack or filenames. Creating now...')
                stack_tifs(ann_folder)

            if not ann_var_arr_exists:
                print('\tMissing array. Creating now...')
                tif_stack2array(ann_folder)

            if orient:
                if not ann_var_or_arr_exists or not ann_var_or_fnames_exists:
                    print(
                        '\tMissing oriented array or oriented filenames. '\
                        'Creating now...'
                    )
                    stack_orient(ann_folder)
            
            if not ann_var_json_exists and annot_type=='soma':
                print('\tMissing .json file. Creating now...')
                binary2json(
                    path=path,
                    data_set=data_set,
                    data_type=data_type,
                    data_subset=data_subset,
                    subset_type=subset_type,
                    annot_type=annot_type,
                    mode = 'prep',
                )


# If preprocess.py is run directly
if __name__ == '__main__':

    # Run preprocess()
    preprocess()
    