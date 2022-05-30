import os
import sys
import cv2
import json
import time
from cv2 import merge
import torch
import shutil
import random
import numpy            as np
import subprocess       as sp
from GPUtil             import getAvailable
from pycocotools    import coco



class HiddenPrints:
    """Hides print output"""

    def __enter__(self):
        """Upon entering"""

        # Save stdout parameters
        self._original_stdout = sys.stdout

        # Replace stdout with null
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Upon leaving"""

        # Close stdout replacement
        sys.stdout.close()

        # Restore original stdout parameters
        sys.stdout = self._original_stdout


def path_gen(elmnts, file=False):
    """Generates path string out of elements
    
    Args:
        elmnts (list of strings): path elements
        file (bool): whether last element is a file
 
    Returns:
        (string) merged path string
    """

    # Initialize path string
    path = ''

    # Append elements
    for elmnt in elmnts:
        path += f'{elmnt}/'
    
    # Remove last forward slash if file
    if file:
        path = path[:-1:]

    return path


def make_data_subset(

    # Data
    path                    = '/mnt/sdg/maxs',
    annot_type              = 'soma',

    # Source 
    source_data_set         = 'LIVECell',
    source_data_type        = 'full_set',
    source_data_subset      = 'full_set',
    source_subset_type      = 'all',

    # Target 
    target_data_set         = 'LIVECell',
    target_data_type        = 'part_set',
    target_data_subset      = '5',
    target_subset_type      = 'val',
    
    # Exclude 1 
    exclude                 = True,
    exclude_data_set_1      = 'LIVECell',
    exclude_data_type_1     = 'part_set',
    exclude_data_subset_1   = '5',
    exclude_subset_type_1   = 'train',

    # Exclude 2 
    exclude_data_set_2      = 'LIVECell',
    exclude_data_type_2     = 'part_set',
    exclude_data_subset_2   = '5',
    exclude_subset_type_2   = 'test',

    # Other
    ratio                   = .015,
    enrichment              = 1/3,
):

    # Generate folder path strings
    source_image_folder = path_gen([
        path, 
        'data',
        source_data_set,
        source_data_type,
        source_data_subset,
        'images',
        source_subset_type,
    ])
    source_annot_folder = path_gen([
        path, 
        'data',
        source_data_set,
        source_data_type,
        source_data_subset,
        'annotations',
        source_subset_type,
        annot_type,
    ])
    target_image_folder = path_gen([
        path, 
        'data',
        target_data_set,
        target_data_type,
        target_data_subset,
        'images',
        target_subset_type,
    ])
    target_annot_folder = path_gen([
        path, 
        'data',
        target_data_set,
        target_data_type,
        target_data_subset,
        'annotations',
        target_subset_type,
        annot_type,
    ])
    exclude_image_folder_1 = path_gen([
        path, 
        'data',
        exclude_data_set_1,
        exclude_data_type_1,
        exclude_data_subset_1,
        'images',
        exclude_subset_type_1,
    ])
    exclude_image_folder_2 = path_gen([
        path, 
        'data',
        exclude_data_set_2,
        exclude_data_type_2,
        exclude_data_subset_2,
        'images',
        exclude_subset_type_2,
    ])

    filenames = os.listdir(f'{source_image_folder}')
    image_names = [filename for filename in filenames if filename.endswith('.tif')]
    selected_names = random.sample(image_names, k=round((1-enrichment)*ratio*len(image_names)))
    extra_names = random.sample(list(set(image_names) - set(selected_names)), k=round(enrichment*ratio*len(image_names)))
    selected_names.extend(extra_names)
    selected_names = sorted(list(set(selected_names)))

    if exclude:
        exclude_filenames_1 = os.listdir(f'{exclude_image_folder_1}')
        exclude_filenames_2 = os.listdir(f'{exclude_image_folder_2}')
        selected_names = [dataset_name for dataset_name in selected_names if dataset_name not in exclude_filenames_1 and dataset_name not in exclude_filenames_2]

    # Create output directories if missing
    if not os.path.isdir(f'{target_image_folder}'):
        os.makedirs(f'{target_image_folder}')
    if not os.path.isdir(f'{target_annot_folder}'):
        os.makedirs(f'{target_annot_folder}')

    for filename in selected_names:
        shutil.copyfile(
            f'{source_image_folder}{filename}', 
            f'{target_image_folder}{filename}'
        )
        shutil.copyfile(
            f'{source_annot_folder}{filename}', 
            f'{target_annot_folder}{filename}'
        )


def remove_bad_annotations(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_type = 'per_celltype',
    data_subset = 'A172',
    subset_type = 'train'
):

    # Generate folder path strings
    image_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'images',
        subset_type
    ])
    annot_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'annotations',
        subset_type
    ])
    LIVECell_folder = path_gen([
        path,
        'data',
        data_set,
    ])

    print('\tRemoving one cell and weird files from both the image and annotation folder...')

    one_cell_annotation_files = []
    with open(f'{LIVECell_folder}one_cell_annotations.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            one_cell_annotation_files.append(f'{args[0]}.tif')

    weird_annotation_files = []
    with open(f'{LIVECell_folder}weird_annotations.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            weird_annotation_files.append(f'{args[0]}.tif')
    
    for file in one_cell_annotation_files:
        try:
            os.remove(f'{image_folder}{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}mem/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}thick_mem/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}soma/{file}')
        except OSError:
            pass

    for file in weird_annotation_files:
        try:
            os.remove(f'{image_folder}{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}mem/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}thick_mem/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}soma/{file}')
        except OSError:
            pass

    print('\tDone')


def get_gpu_memory():
    """Gets used GPU memory
    
    Returns:
        memory_use_values (list): GPU memory usage in MB 
            for all GPU devices"""

    # Create output separation lambda
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    # Define command for memory retrieval
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    # Get memory
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    #Format output
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]

    return memory_use_values


def optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        if torch.is_tensor(param):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if torch.is_tensor(subparam):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                      
                        
def map_tensor_coordinates(input, coordinates, chan_dim, batch_dim):
    """ PyTorch version of scipy.ndimage.interpolation.map_coordinates
    
    Args:
        input (BxCxHxW tensor): tensor to be distorted
        coordinates (2x<FlatInput> tensor): new coordinates of all
            data points
        chan_dim (1x<FlatInput> tensor): index tensor for channel
            dimension        
        batch_dim (1x<FlatInput> tensor): index tensor for batch
            dimension
    """

    # Get input shape
    h = input.shape[2]
    w = input.shape[3]

    # Wrap coordinates around image
    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    # Get nearest pixels
    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()

    # Calculate distortion magnitudes
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())

    # Wrap coordinates around image
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    # Process distortions through input
    f00 = input[batch_dim, chan_dim, co_floor[0], co_floor[1]]
    f10 = input[batch_dim, chan_dim, co_floor[0], co_ceil[1]]
    f01 = input[batch_dim, chan_dim, co_ceil[0], co_floor[1]]
    f11 = input[batch_dim, chan_dim, co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)

    return fx1 + d2 * (fx2 - fx1)  


def reset_seeds():
    """Resets all random seeds
    
    Args:
        seed_offset (int): value to offset initial seed"""

    # Generate a new seed
    new_seed = torch.seed()
    if new_seed >= 2**32: 
        new_seed = new_seed % 2**32

    # Reset all randomness
    os.environ['PYTHONHASHSEED'] = str(new_seed)
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)


def get_gpu_list(
    n_gpus,
    kill_my_gpus,
    reserved_gpus,
    gpu_check_duration,
    gpu_usage_limit,
):
    """Retrieves list of freely usable GPUs
    
    Args:
        n_gpus (int): number of GPUs needed
        kill_my_gpus (bool): whether to kill GPUs 2 and 3
        reserved_gpus (list): list of reserved device IDs
        gpu_check_duration (int): GPU observation time in seconds
        gpu_usage_limit (int): free memory required on each GPU in MB
    """

    # Kill processes on GPUs 0 to 3
    if kill_my_gpus:
        os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 0 && $2 <= 3 {print $5}')""")
        print('')

    # Get unavailable GPUs
    gpu_usage = np.zeros((gpu_check_duration, 10), dtype='uint16')
    for s in range(gpu_check_duration):
        sys.stdout.write('\r')
        sys.stdout.write(
            f'\tChecking which GPUs are available for {gpu_check_duration-s} seconds...'
        )
        sys.stdout.flush()
        gpu_usage[s, :] = get_gpu_memory()
        time.sleep(1)
    max_gpu_usage = np.amax(gpu_usage, axis=0)
    hidden_used_gpus = [i for i, x in enumerate(max_gpu_usage > gpu_usage_limit) if x]

    # Get possibly available GPUs
    possibly_available_gpus = getAvailable(
        limit=100, 
        memoryFree=gpu_usage_limit
    )
    
    # Get available GPUs
    gpu_device_ids = [
        gpu 
        for gpu in possibly_available_gpus
        if gpu not in hidden_used_gpus and gpu not in reserved_gpus
    ]

    if len(gpu_device_ids) < n_gpus:
        raise RuntimeError('Too few GPUs available')

    return gpu_device_ids[:n_gpus]


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    #cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def enlarged_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_norm = np.where(cnt_norm < 0, cnt_norm-1, cnt_norm)
    cnt_norm = np.where(cnt_norm > 0, cnt_norm+1, cnt_norm)
    cnt_enlarged = cnt_norm + [cx, cy]
    cnt_enlarged = cnt_enlarged.astype(np.int32)

    return cnt_enlarged

def separate_train_val():
    # get names from celltype/annotations/train/soma 
    celltypes = ['A172', 'BV2', 'BT474', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']
    for celltype in celltypes:
        train_path=f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/annotations/train/soma'
        train_filenames = os.listdir(train_path)
        train_names = [filename for filename in train_filenames if filename.endswith('.tif')]
        train_names.sort()

        files_in_val_folder = os.listdir(f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/images/val')
        images_in_val_folder = [filename for filename in files_in_val_folder if filename.endswith('.tif')]
        for train_name in train_names:
            if train_name in images_in_val_folder:
                os.remove(f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/images/val/{train_name}')
        
        val_path=f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/annotations/val/soma'
        val_filenames = os.listdir(val_path)
        val_names = [filename for filename in val_filenames if filename.endswith('.tif')]
        val_names.sort()

        files_in_train_folder = os.listdir(f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/images/train')
        images_in_train_folder = [filename for filename in files_in_train_folder if filename.endswith('.tif')]
        for val_name in val_names:
            if val_name in images_in_train_folder:
                os.remove(f'/mnt/sdg/maxs/data/LIVECell/per_celltype/{celltype}/images/train/{val_name}')

def merge_json():

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

    with open('/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/train/soma/variables/filenames.txt', 'r') as infile:
        train_filenames = []
        for line in infile:
            train_filenames.append(line.split('\n')[0])
    with open('/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/test/soma/variables/filenames.txt', 'r') as infile:
        test_filenames = []
        for line in infile:
            test_filenames.append(line.split('\n')[0])
    with open('/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/val/soma/variables/filenames.txt', 'r') as infile:
        val_filenames = []
        for line in infile:
            val_filenames.append(line.split('\n')[0])
    with open('/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/extra/soma/variables/filenames.txt', 'r') as infile:
        ext_filenames = []
        for line in infile:
            ext_filenames.append(line.split('\n')[0])
            
    # Load .json files
    test_json_file = coco.COCO(
        '/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/test.json'
    )
    train_json_file = coco.COCO(
        '/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/train.json'
    )
    val_json_file = coco.COCO(
        '/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/val.json'
    )

    # Get image IDs
    train_img_ids = train_json_file.getImgIds()
    test_img_ids = test_json_file.getImgIds()
    val_img_ids = val_json_file.getImgIds()

    # Get image data
    train_imgs = train_json_file.loadImgs(train_img_ids) 
    test_imgs = test_json_file.loadImgs(test_img_ids) 
    val_imgs = val_json_file.loadImgs(val_img_ids) 

    # Merge all images
    test_imgs.extend(val_imgs)
    train_imgs.extend(test_imgs)

    # Remove double images
    image_ids = []
    for train_img in train_imgs:
        if train_img not in images and (train_img['file_name'] in train_filenames or train_img['file_name'] in test_filenames or train_img['file_name'] in val_filenames):
            images.append(train_img)
            image_ids.append(train_img['id'])

    # Load all annotations in the current image
    for iI in range(len(image_ids)):
        if image_ids[iI] in train_img_ids:
            train_ann_ids_image = train_json_file.getAnnIds(imgIds = image_ids[iI])
            train_anns_image = train_json_file.loadAnns(train_ann_ids_image) 
            for trains_ann_image in train_anns_image:
                annotations.append(trains_ann_image)    
        elif image_ids[iI] in test_img_ids:
            test_ann_ids_image = test_json_file.getAnnIds(imgIds = image_ids[iI])
            test_anns_image = test_json_file.loadAnns(test_ann_ids_image)  
            for test_ann_image in test_anns_image:
                annotations.append(test_ann_image)  
        elif image_ids[iI] in val_img_ids:
            val_ann_ids_image = val_json_file.getAnnIds(imgIds = image_ids[iI])
            val_anns_image = val_json_file.loadAnns(val_ann_ids_image)  
            for val_ann_image in val_anns_image:
                annotations.append(val_ann_image)  

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
    with open('/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/all.json', 'w') as outfile:
        json.dump(json_file, outfile)

def make_json(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_type = 'part_set',
    data_subset = '5',
    subset_type = 'val'
):

    annot_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'annotations',
        subset_type,
        'soma',
        'variables'
    ])
    output_folder = path_gen([
        path,
        'data',
        data_set,
        data_type,
        data_subset,
        'annotations',
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

    with open(f'{annot_folder}filenames.txt', 'r') as infile:
        filenames = []
        for line in infile:
            filenames.append(line.split('\n')[0])
            
    # Load .json files
    all_json_file = coco.COCO(
        '/mnt/sdg/maxs/data/LIVECell/full_set/full_set/annotations/all.json'
    )

    # Get image IDs
    all_img_ids = all_json_file.getImgIds()

    # Get image data
    all_imgs = all_json_file.loadImgs(all_img_ids) 

    # Remove double images
    image_ids = []
    for all_img in all_imgs:
        if all_img['file_name'] in filenames:
            images.append(all_img)
            image_ids.append(all_img['id'])

    # Load all annotations in the current image
    for iI in range(len(image_ids)):
        ann_ids_image = all_json_file.getAnnIds(imgIds = image_ids[iI])
        anns_image = all_json_file.loadAnns(ann_ids_image) 
        for ann_image in anns_image:
            annotations.append(ann_image)    

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
    with open(f'{output_folder}{subset_type}.json', 'w') as outfile:
        json.dump(json_file, outfile)
