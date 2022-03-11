import os
import sys
import cv2
import json
import numpy                    as np
from utils                      import path_gen
from pycocotools                import coco
from pycocotools                import mask

def binary2json(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'extra',
    print_separator = '$',
    model = '1',
    snapshot = 50
):
    """Converts binary predictions to .json COCO
        instance segmentation format

    Note:
        Category information
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
        snapshot (int): training epoch age of snapshot used
            for evaluation (default = 50)
 
    Returns:
        Converted .json COCO instance segmentation file in
            the same folder as the binary array
    """
    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 70, '\n', sep='')
    print(f'\tConverting binary predictions to .json COCO format...')
    
    # Generate folder path strings
    data_folder = path_gen([
        path,
        'data',
        data_set,
        'images',
        data_subset,
        'variables'
    ])
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_subset,
        model,
        'deployment'
    ])

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

    # Load binary prediction data with its identifiers
    arr = np.load(f'{results_folder}FusionNet_snapshot{snapshot}_prediction_array.npy')
    with open(f'{data_folder}filenames.txt', 'r') as infile:
        filenames = [line.split('\n')[0] for line in infile]

    # Fill in the 'images' and 'annotations' lists
    annotation_id = 0
    for image_id, filename in enumerate(filenames, 1):

        # Display progress
        image_ratio = (image_id - 1) / (len(filenames) - 1)
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
            if contour.size >= 6:

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
                    'category_id': 1,
                    'segmentation': [segmentation],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd' : 0
                }
                annotations.append(annotation)

    # Parse everything into .json file structure
    predictions = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'info': info,
        'licenses': licenses
    }

    # Save .json file
    with open(f'{results_folder}FusionNet_snapshot{snapshot}_predictions.json', 'w') as outfile:
        json.dump(predictions, outfile)


def evaluate(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'train',
    print_separator = '$',
    model = '1',
    snapshot = 50   
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
                                <data_subset>.json
                results/
                    <data_set>/<data_subset>/
                        <model>/deployment/
                            predictions.json
    Args:
        path (string): path to training data folder
        data_set (string): training data set
        data_subset (string): training data subset
        print_separator (string): print output separation
            character (default = '$')
        model (string): current model identifier (default = 1)
        snapshot (int): training epoch age of snapshot used
            for evaluation (default = 50)

    Returns:

    """
    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 87, '\n', sep='')

    # Generate path strings
    gt_filepath = path_gen([
        path,
        'models',
        data_set,
        'annotations',
        data_subset,
        f'{data_subset}.json'
    ], file=True)
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_subset,
        model,
        'deployment'
    ])

    # Load ground truth and detection .json data
    gt_json_file = coco.COCO(gt_filepath)       
    dt_json_file = coco.COCO(
        f'{results_folder}FusionNet_snapshot{snapshot}_predictions.json'
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
    n_ap_steps = 10,
    n_cellsizes = 4

    # Initialize detection performance dictionary
    performance = {
            'TP': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int),
            'FP': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int),
            'FN': np.zeros((n_cellsizes, n_ap_steps, dt_n_images), dtype=int)
    }

    # Calculate detection performance for all processed images and instances
    for iI in range(dt_n_images):

        # Load all annotations in the current image
        dt_ann_ids_image = dt_json_file.getAnnIds(imgIds = dt_img_ids[iI])
        dt_anns_image = dt_json_file.loadAnns(dt_ann_ids_image)        
        gt_ann_ids_image = gt_json_file.getAnnIds(imgIds = dt_img_ids[(iI+1) % dt_n_images])
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
        all_ious = mask.iou(dt_rle, gt_rle, [0])  

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
    ap = np.mean(ap_iou, axis=1)
    ar_iou = np.zeros((n_cellsizes, n_ap_steps), dtype=float)
    for entry in range(1, dt_n_images+1):
        ar_iou += (abs_precision[:, :, entry] - abs_precision[:, :, entry-1]) * cumul_interp_recall[:, :, entry]
    afnr_iou = 1 - ar_iou
    afnr = np.mean(afnr_iou, axis=1)
    af1_iou = 2 * ap_iou * ar_iou / (ap_iou + ar_iou) 
    af1 = np.mean(af1_iou, axis=1)

    with open(f'{results_folder}FusionNet_snapshot{snapshot}_performance_metrics.txt', 'w') as outfile:
        args = f'{ap_iou},{ap},{afnr_iou},{afnr},{af1_iou},{af1}'
        outfile.write(args)


# If eval.py is run directly
if __name__ == '__main__':
    
    # Convert predictions
    binary2json()

    # Evaluate predictions
    evaluate()
    print('\n', end='')