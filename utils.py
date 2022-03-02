import os
import cv2
import sys
import json
import tifffile
import numpy as np
from PIL import Image
from pycocotools import coco
from pycocotools import mask


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def path_gen(elmnts):
    path = ''
    for elmnt in elmnts:
        path += f'{elmnt}/'
    return path

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
            out_file_line = f'{orig_filename},{imgIds[iImage]}\n'
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


def compute_IoU(gt_filepath, dt_filepath):
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

    n_ap_steps = 10
    n_cellsizes = 4

    gt_json_file = coco.COCO(gt_filepath)       ;   dt_json_file = coco.COCO(dt_filepath)
    gt_img_ids = gt_json_file.getImgIds()       ;   dt_img_ids = dt_json_file.getImgIds()
    gt_imgs = gt_json_file.loadImgs(gt_img_ids) ;   dt_imgs = dt_json_file.loadImgs(dt_img_ids) 
    gt_n_images = len(gt_img_ids)               ;   dt_n_images = len(dt_img_ids) 

    if gt_n_images != dt_n_images:
        raise ValueError
    else:
        n_images = dt_n_images

    out_folder = f'{gt_filepath.rsplit("/", 1)[0]}/' 
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    
    performance = {
            'TP': np.zeros((n_cellsizes, n_ap_steps, n_images), dtype=int),
            'FP': np.zeros((n_cellsizes, n_ap_steps, n_images), dtype=int),
            'FN': np.zeros((n_cellsizes, n_ap_steps, n_images), dtype=int)
    }

    for i_image in range(n_images):
        dt_ann_ids_image = dt_json_file.getAnnIds(imgIds = dt_img_ids[i_image])
        dt_anns_image = dt_json_file.loadAnns(dt_ann_ids_image)        
        gt_ann_ids_image = gt_json_file.getAnnIds(imgIds = dt_img_ids[(i_image+1) % n_images])
        gt_anns_image = gt_json_file.loadAnns(gt_ann_ids_image)

        dt_area = np.array([dt_anns_image[i_dt_anns]['area'] for i_dt_anns in range(len(dt_anns_image))])
        gt_area = np.array([gt_anns_image[i_gt_anns]['area'] for i_gt_anns in range(len(gt_anns_image))])

        dt_320 = np.transpose(np.tile(dt_area > 500, (len(gt_area), 1)))
        dt_970 = np.transpose(np.tile(dt_area > 1500, (len(gt_area), 1)))
        gt_320 = np.tile(gt_area > 500, (len(dt_area), 1))
        gt_970 = np.tile(gt_area > 1500, (len(dt_area), 1))

        dt_area_map = np.uint8(dt_320) + np.uint8(dt_970) # 0: <320 micrometers, 1, 2: >970 
        gt_area_map = np.uint8(gt_320) + np.uint8(gt_970) 
        dt_gt_area_map = np.uint8(np.logical_and(dt_320, gt_320)) + np.uint8(np.logical_and(dt_970, gt_970))

        dt_rle = [dt_json_file.annToRLE(dt_anns_image[i_dt_anns]) for i_dt_anns in range(len(dt_anns_image))]
        gt_rle = [gt_json_file.annToRLE(gt_anns_image[i_gt_anns]) for i_gt_anns in range(len(gt_anns_image))]

        all_ious = mask.iou(dt_rle, gt_rle, [0])  

        ops = {
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b
        }
        cellsizes = [0, 0, 1, 2]
        comps = ['>=', '==', '==', '==']

        for iou_threshold, ap_step in zip(np.arange(.5, 1,.05), range(n_ap_steps)):
            valid_ious = all_ious >= iou_threshold
            for cellsize in range(4):
                performance['TP'][cellsize, ap_step, i_image] = sum(np.any(np.logical_and(valid_ious, ops[comps[cellsize]](dt_gt_area_map, cellsizes[cellsize])), 1))
                performance['FP'][cellsize, ap_step, i_image] = sum(np.all(np.logical_and(np.logical_not(valid_ious), ops[comps[cellsize]](dt_area_map, cellsizes[cellsize])), 1))
                performance['FN'][cellsize, ap_step, i_image] = sum(np.all(np.logical_and(np.logical_not(valid_ious), ops[comps[cellsize]](gt_area_map, cellsizes[cellsize])), 0))

        j = (i_image + 1) / n_images
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()

    cumul_TP = np.cumsum(performance['TP'], axis=2)
    cumul_FP = np.cumsum(performance['FP'], axis=2)
    cumul_FN = np.cumsum(performance['FN'], axis=2)

    rolled_cumul_FP = np.zeros_like(cumul_FP.shape)
    rolled_cumul_FP = np.roll(cumul_FP, 1, axis = 2)
    rolled_cumul_FP[:, :, 0] = np.zeros((n_cellsizes, n_ap_steps))
    rolled_cumul_FN = np.zeros_like(cumul_FN.shape)
    rolled_cumul_FN = np.roll(cumul_FN, 1, axis = 2)
    rolled_cumul_FN[:, :, 0] = np.zeros((n_cellsizes, n_ap_steps))

    cumul_precision = np.divide(cumul_TP, (cumul_TP + rolled_cumul_FP), out=np.zeros(cumul_TP.shape, dtype=float), where=(cumul_TP + rolled_cumul_FP)!=0)
    cumul_precision = np.insert(cumul_precision, 0, 1., axis=2)
    cumul_recall = np.divide(cumul_TP, (cumul_TP + rolled_cumul_FN), out=np.zeros(cumul_TP.shape, dtype=float), where=(cumul_TP + rolled_cumul_FN)!=0)
    cumul_recall = np.insert(cumul_recall, 0, 1., axis=2)

    cumul_interp_precision = np.zeros((n_cellsizes, n_ap_steps, n_images+1), dtype=float)
    for entry in range(n_images+1): 
        cumul_interp_precision[:, :, entry] = np.amax(cumul_precision[:, :, entry::], axis=2)
    cumul_interp_recall = np.zeros((n_cellsizes, n_ap_steps, n_images+1), dtype=float)
    for entry in range(n_images+1): 
        cumul_interp_recall[:, :, entry] = np.amax(cumul_recall[:, :, entry::], axis=2)

    abs_precision = cumul_TP / (
        cumul_TP[:, :, -1] + 
        cumul_FP[:, :, -1]
    )[:, :, None]
    abs_precision = np.insert(abs_precision, 0, 0., axis=2)
    abs_recall = cumul_TP / (
        cumul_TP[:, :, -1] + 
        cumul_FN[:, :, -1]
    )[:, :, None]
    abs_recall = np.insert(abs_recall, 0, 0., axis=2)

    ap_iou = np.zeros((n_cellsizes, n_ap_steps), dtype=float)
    for entry in range(1, n_images+1):
        ap_iou += (abs_recall[:, :, entry] - abs_recall[:, :, entry-1]) * cumul_interp_precision[:, :, entry]
    ap = np.mean(ap_iou, axis=1)
    ar_iou = np.zeros((n_cellsizes, n_ap_steps), dtype=float)
    for entry in range(1, n_images+1):
        ar_iou += (abs_precision[:, :, entry] - abs_precision[:, :, entry-1]) * cumul_interp_recall[:, :, entry]
    afnr_iou = 1 - ar_iou
    afnr = np.mean(afnr_iou, axis=1)
    af1_iou = 2 * ap_iou * ar_iou / (ap_iou + ar_iou) 
    af1 = np.mean(af1_iou, axis=1)

    return ap_iou, ap, afnr_iou, afnr, af1_iou, af1


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

def binary2json(dir):
    # Define which colors match which categories in the images
    # category_ids = {
    #     1: {
    #         255: 'A172'
    #     }
    # }

    image_id = 0
    annotation_id = 0
    iscrowd = 0
    category_id = 1

    images = []
    annotations = []
    categories = [
        {
            'supercategory': 'cell',
            'id': 1,
            'name': 'cell'
        }
    ]
    info = {
        'year': 2022,
        'version': 0.1,
        'description': 'LIVECell 2022 Predictions',
        'contributor': 'Max Levian Sterling',
        'date_created': '2022/02/25'
    }
    licenses = [
        {
            'id': 1, 
            'name': 'Attribution-NonCommercial 4.0 International License', 
            'url': 'https://creativecommons.org/licenses/by-nc/4.0/'
        }
    ]

    out_folder = f'{dir}variables/'
    arr = np.load(f'{out_folder}array.npy') #predicted_array.npy
    filenames = []
    with open(f'{out_folder}filenames.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            filenames.append(args[0])

    for page, filename in zip(range(arr.shape[0]), filenames):
        print('.', end='')
        image_id += 1

        image = {
            'id': image_id,
            'width': arr.shape[2],
            'height': arr.shape[1],
            'file_name': filename,
        }
        images.append(image)

        contours, _ = cv2.findContours(arr[page,:,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:

                annotation_id += 1
                segmentation = contour.astype(float).flatten().tolist()
                x = segmentation[0::2]
                y = segmentation[1::2]
                area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
                min_x = min(x); max_x = max(x)
                min_y = min(y); max_y = max(y)
                bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'segmentation': [segmentation],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd' : iscrowd
                }
                annotations.append(annotation)

    predictions = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'info': info,
        'licenses': licenses
    }

    with open(f'{out_folder}predictions.json', 'w') as outfile:
        json.dump(predictions, outfile)
    #print(json.dumps(predictions))


compute_IoU(r'C:/Users/Max-S/tndrg/Data/LIVECell/annotations/val.json', r'C:/Users/Max-S/tndrg/Data/LIVECell/annotations/val.json')