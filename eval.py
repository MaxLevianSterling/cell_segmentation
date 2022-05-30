import os
import sys
import cv2
import json
import numpy        as np
from utils          import path_gen
from utils          import HiddenPrints
from utils          import scale_contour
from utils          import enlarged_contour
from pycocotools    import coco
from pycocotools    import mask
from math           import ceil


def binary2json(
    path,
    data_set,
    data_type,
    data_subset,
    subset_type,
    annot_type,
    mode,
    model = '',
    checkpoint = '',
    print_separator = '$',
):
    """Converts binary predictions to .json COCO
        instance segmentation format

    Note:   
        Assumes data represent monocultures
        Required folder structure:
            <path>/
                results/
                    <data_set>/<data_subset>/
                        <model>/deployment/

    Args:
        path (string): path to training data folder
        data_set (string): training data set
        data_subset (string): training data subset
        print_separator (string): print output separation
            character (default = '$')
        model (string): current model identifier (default = 1)
        checkpoint (int): training epoch age of checkpoint used
            for evaluation (default = 50)
        mode (string): function mode can be 'eval' or 'prep',
            depending on if the input or output mask array 
            is used to generate the .json file
 
    Returns:
        Converted .json COCO instance segmentation file in
            the same folder as the binary array
    """
    
    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 71, '\n', sep='')
    print(f'\tConverting binary masks to .json COCO format...')
    
    # Generate folder path strings
    image_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'images',
        subset_type,
    ])
    annot_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'annotations',
        subset_type,
        annot_type,
    ])
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_type,
        data_subset,
        subset_type,
        annot_type,
        model,
        'deployment'
    ])
    
    # Skip if .json file exists already
    if mode == 'eval':
        json_exists = os.path.isfile(
            f'{results_folder}checkpoint{checkpoint}_predictions.json'
        )
        if json_exists:
            print('\tFile already existed. Continuing...')
            return
            
    # Load binary mask array and filename identifiers
    if mode == 'eval':
        arr = np.load(
            f'{results_folder}checkpoint{checkpoint}_predictions.npy'
        )
        with open(f'{image_folder}variables/filenames.txt', 'r') as infile:
            filenames = []
            cat_ids = []
            for line in infile:
                filenames.append(line.split('\n')[0])
                cat_ids.append(line.split('_')[0])
    elif mode == 'prep':
        arr = np.load(
            f'{annot_folder}variables/array.npy'
        )
        with open(f'{annot_folder}variables/filenames.txt', 'r') as infile:
            filenames = []
            cat_ids = []
            for line in infile:
                filenames.append(line.split('\n')[0])
                cat_ids.append(line.split('_')[0])

    # Define cell type category identities:
    cat_keys = {
        'A172': 2,
        'BT474': 4,
        'BV2': 3,
        'Huh7': 5,
        'MCF7': 6,
        'SHSY5Y': 1,
        'SkBr3': 7,
        'SKOV3': 8
    }
    
    # Codify category IDs
    cat_ids = [
        cat_keys[cat] 
        for cat in cat_ids
    ]

    # Initiate .json file structure
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

    # Fill in the 'images' and 'annotations'
    annotation_id = 0
    for image_id, filename in enumerate(filenames, 1):

        # Display progress
        image_ratio = (image_id - 1) / (len(filenames))
        sys.stdout.write('\r')
        sys.stdout.write(
            "\tImages: [{:<{}}] {:.0f}%".format(
                "=" * int(20*image_ratio), 20, 100*image_ratio,
            )
        )
        sys.stdout.flush()

        # Fill in the 'images' list
        image = {
            'id': image_id,
            'width': arr.shape[2],
            'height': arr.shape[1],
            'file_name': filename,
        }
        images.append(image)

        # Find all cell contours in the image
        contours, _ = cv2.findContours(
            arr[image_id-1, :, :], 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Run past each contour
        for contour in contours:
            
            # Check if polygon is 2D
            if cv2.moments(contour)['m00'] > 0:
                
                (x,y),radius = cv2.minEnclosingCircle(contour)
                radius = ceil(radius)
                contour = enlarged_contour(contour, (radius+1)/radius)
                
                # Get cell contour data
                annotation_id += 1
                segmentation = contour.astype(float).flatten().tolist()
                x = segmentation[0::2]
                y = segmentation[1::2]
                area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
                min_x = min(x); max_x = max(x)
                min_y = min(y); max_y = max(y)
                bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
                
                # Parse in the annotation data
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': cat_ids[image_id-1],
                    'segmentation': [segmentation],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd' : 0
                }
                annotations.append(annotation)

    # Parse everything into .json file structure
    json_file = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'info': info,
        'licenses': licenses
    }

    # Display status
    print('\n\tSaving .json file...', end='')

    # Save .json file
    if mode == 'eval':
        with open(f'{results_folder}checkpoint{checkpoint}_predictions.json', 'w') as outfile:
            json.dump(json_file, outfile)
    elif mode == 'prep':
        with open(f'{annot_folder}variables/json.json', 'w') as outfile:
            json.dump(json_file, outfile)

    print('\n')


def evaluate(
    path,
    data_set,
    data_type,
    data_subset,
    subset_type,
    annot_type,
    mode,
    model,
    checkpoint,
    print_separator = '$',
):
    """Evaluates cell instance segmentation network output with 
        AP, AFNR, and F1 scores
    
    Note:
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        annotations/
                            <data_subset>
                                <.json file>
                results/
                    <data_set>/<data_subset>/
                        <model>/deployment/
                            <.json file>

    Args:
        path (string): path to training data folder
        data_set (string): training data set
        data_subset (string): training data subset
        print_separator (string): print output separation
            character
        model (string): current model identifier
        checkpoint (int): training epoch age of checkpoint used
            for evaluation
        mode (string): function mode can be 'original' or 
            'from_bin', depending on if the .json file used 
            is the original or generated from binary masks 
            
    Returns:
        A log of all performance metrics in the results 
            folder
    """

    # Being beautiful is not a crime
    print('\n\t', f'{print_separator}' * 55, '\n', sep='')

    # Generate ground truth path string
    if mode == 'original':
        gt_filepath = path_gen([
            path, 
            'data',
            data_set,
            data_type,
            data_subset,
            'annotations',
            f'{subset_type}.json'
        ], file=True)
    elif mode == 'from_bin':
        gt_filepath = path_gen([
            path, 
            'data',
            data_set,
            data_type,
            data_subset,
            'annotations',
            subset_type,
            annot_type,
            'variables'
            'json.json'
        ], file=True)

    # Generate result folder path string
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_type,
        data_subset,
        subset_type,
        annot_type,
        model,
        'deployment'
    ])

    # Load ground truth and detection .json data
    with HiddenPrints():
        gt_json_file = coco.COCO(gt_filepath)       
        dt_json_file = coco.COCO(
           f'{results_folder}checkpoint{checkpoint}_predictions.json'
        )

    # Get image IDs
    gt_img_ids = gt_json_file.getImgIds()  
    dt_img_ids = dt_json_file.getImgIds()

    # Get image data
    gt_imgs = gt_json_file.loadImgs(gt_img_ids)
    dt_imgs = dt_json_file.loadImgs(dt_img_ids) 

    # Get number of ground truth and detection images
    gt_n_images = len(gt_img_ids)
    dt_n_images = len(dt_img_ids) 

    # Initialize COCO/LIVECell parameters     
    n_ap_steps = 10
    n_cellsizes = 4

    # Initialize detection performance dictionary
    performance = {
            'TP': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int),
            'FP': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int),
            'FN': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int)
    }

    # Calculate detection performance for all processed images and instances
    for iI in range(dt_n_images):

        # Get ground truth image ID equivalent 
        filename = dt_imgs[iI]['file_name']
        for gt_img in gt_imgs:
            if gt_img['file_name'] == filename:
                gt_id = gt_img['id']
                break

        # Load all annotations in the current image
        dt_ann_ids_image = dt_json_file.getAnnIds(imgIds = dt_img_ids[iI])
        dt_anns_image = dt_json_file.loadAnns(dt_ann_ids_image)        
        gt_ann_ids_image = gt_json_file.getAnnIds(imgIds = gt_id)
        gt_anns_image = gt_json_file.loadAnns(gt_ann_ids_image)

        # Get all ground truth and detected instance areas
        dt_area = np.array([dt_anns_image[i_dt_anns]['area'] for i_dt_anns in range(len(dt_anns_image))])
        gt_area = np.array([gt_anns_image[i_gt_anns]['area'] for i_gt_anns in range(len(gt_anns_image))])

        # Construct categorical instance size maps
        dt_320 = np.transpose(np.tile(dt_area > 500, (len(gt_area), 1)))
        dt_970 = np.transpose(np.tile(dt_area > 1500, (len(gt_area), 1)))
        gt_320 = np.tile(gt_area > 500, (len(dt_area), 1))
        gt_970 = np.tile(gt_area > 1500, (len(dt_area), 1))

        # Merge categorical instance size maps (0:<320 micrometers; 1; 2:>970) 
        dt_area_map = np.uint8(dt_320) + np.uint8(dt_970) 
        gt_area_map = np.uint8(gt_320) + np.uint8(gt_970) 
        dt_gt_area_map = np.uint8(np.logical_and(dt_320, gt_320)) + np.uint8(np.logical_and(dt_970, gt_970))

        # Convert instance segmentation format to compressed RLE
        dt_rle = [dt_json_file.annToRLE(dt_anns_image[i_dt_anns]) for i_dt_anns in range(len(dt_anns_image))]
        gt_rle = [gt_json_file.annToRLE(gt_anns_image[i_gt_anns]) for i_gt_anns in range(len(gt_anns_image))]

        # Get the IoUs of all detections crossed with all ground truth instances
        all_ious = mask.iou(dt_rle, gt_rle, [0 for gt in range(len(gt_rle))])  

        # Define loop generalisers
        ops = {
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b
        }
        cellsizes = [0, 0, 1, 2]
        comps = ['>=', '==', '==', '==']

        # Calculate detection performance for all instances
        for ap_step, iou_threshold in enumerate(np.arange(.5, 1,.05)):
            valid_ious = all_ious >= iou_threshold
            for cellsize in range(n_cellsizes):
                performance['TP'][cellsize, ap_step, iI] = sum(np.any(np.logical_and(valid_ious, ops[comps[cellsize]](dt_gt_area_map, cellsizes[cellsize])), 1))
                performance['FP'][cellsize, ap_step, iI] = sum(np.all(np.logical_and(np.logical_not(valid_ious), ops[comps[cellsize]](dt_area_map, cellsizes[cellsize])), 1))
                performance['FN'][cellsize, ap_step, iI] = sum(np.all(np.logical_and(np.logical_not(valid_ious), ops[comps[cellsize]](gt_area_map, cellsizes[cellsize])), 0))

        # Display progress
        image_ratio = (iI) / (dt_n_images - 1)
        sys.stdout.write('\r')
        sys.stdout.write(
            "\tImages: [{:<{}}] {:.0f}%".format(
                "=" * int(20*image_ratio), 20, 100*image_ratio,
            )
        )
        sys.stdout.flush()

    # Calculate cumulative performance for PR/RP curve
    cumul_TP = np.cumsum(performance['TP'], axis=2)
    cumul_FP = np.cumsum(performance['FP'], axis=2)
    cumul_FN = np.cumsum(performance['FN'], axis=2)

    # Roll false performance metrics to align performance metrics for addition
    rolled_cumul_FP = np.zeros_like(cumul_FP.shape)
    rolled_cumul_FP = np.roll(cumul_FP, 1, axis = 2)
    rolled_cumul_FP[:, :, 0] = np.zeros((n_cellsizes, n_ap_steps))
    rolled_cumul_FN = np.zeros_like(cumul_FN.shape)
    rolled_cumul_FN = np.roll(cumul_FN, 1, axis = 2)
    rolled_cumul_FN[:, :, 0] = np.zeros((n_cellsizes, n_ap_steps))

    # Calculate cumulative precision and recall
    cumul_precision = np.divide(cumul_TP, (cumul_TP + rolled_cumul_FP), out=np.zeros(cumul_TP.shape, dtype=float), where=(cumul_TP + rolled_cumul_FP)!=0)
    cumul_precision = np.insert(cumul_precision, 0, 1., axis=2)
    cumul_recall = np.divide(cumul_TP, (cumul_TP + rolled_cumul_FN), out=np.zeros(cumul_TP.shape, dtype=float), where=(cumul_TP + rolled_cumul_FN)!=0)
    cumul_recall = np.insert(cumul_recall, 0, 1., axis=2)

    # Use interpolated metrics where appropriate
    cumul_interp_precision = np.zeros((n_cellsizes, n_ap_steps, dt_n_images+1), dtype=float)
    for entry in range(dt_n_images+1): 
        cumul_interp_precision[:, :, entry] = np.amax(cumul_precision[:, :, entry::], axis=2)
    cumul_interp_recall = np.zeros((n_cellsizes, n_ap_steps, dt_n_images+1), dtype=float)
    for entry in range(dt_n_images+1): 
        cumul_interp_recall[:, :, entry] = np.amax(cumul_recall[:, :, entry::], axis=2)

    # Calculate denominators for the PR/RP curves
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

    # Calculate final performance metrics
    ap_iou = np.zeros((n_cellsizes, n_ap_steps), dtype=float)
    for entry in range(1, dt_n_images+1):
        ap_iou += (abs_recall[:, :, entry] - abs_recall[:, :, entry-1]) * cumul_interp_precision[:, :, entry]
    ap = np.nanmean(ap_iou, axis=1)
    ar_iou = np.zeros((n_cellsizes, n_ap_steps), dtype=float)
    for entry in range(1, dt_n_images+1):
        ar_iou += (abs_precision[:, :, entry] - abs_precision[:, :, entry-1]) * cumul_interp_recall[:, :, entry]
    afnr_iou = 1 - ar_iou
    afnr = np.nanmean(afnr_iou, axis=1)
    af1_iou = 2 * ap_iou * ar_iou / (ap_iou + ar_iou) 
    af1 = np.nanmean(af1_iou, axis=1)

    with open(f'{results_folder}checkpoint{checkpoint}_performance_metrics.txt', 'w') as outfile:
        args = f'ap_iou{ap_iou}\n\nap{ap}\n\nafnr_iou{afnr_iou}\n\nafnr{afnr}\n\naf1_iou{af1_iou}\n\naf1{af1}'
        outfile.write(args)


# If eval.py is run directly
if __name__ == '__main__':
    
    # Convert predictions if necessary
    binary2json(
        path = '/mnt/sdg/maxs',
        data_set = 'LIVECell',
        data_type = 'per_celltype',
        data_subset = 'BV2',
        subset_type = 'test',
        annot_type = 'soma',
        mode = 'eval',
        model = 'baseline',
        checkpoint = 2000,
        print_separator = '$',
    )
    
    # Evaluate predictions
    evaluate(
        path = '/mnt/sdg/maxs',
        data_set = 'LIVECell',
        data_type = 'per_celltype',
        data_subset = 'BV2',
        subset_type = 'test',
        annot_type = 'soma',
        mode = 'original',
        model = 'baseline',
        checkpoint = 2000,
        print_separator = '$',
    )
    
    print('\n\n', end='')
