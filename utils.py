import os
import sys
import torch
import shutil
import random
import numpy            as np
import subprocess       as sp
from threading          import Thread, Timer
from math               import ceil


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


def unpad(arr, pad_widths):


    """Unpads a numpy array
    
    Args:
        arr (np.array): input array
        pad_widths (int): pad widths to remove
 
    Returns:
        (np.array) unpadded array
    """

    # Initialize array slices
    slices = []

    # For every array dimension
    for c in pad_widths:

        # Take inverse of the far point
        e = None if c[1] == 0 else -c[1]

        # Create slice object for that dimension
        slices.append(slice(c[0], e))
    
    # Unpad
    return arr[tuple(slices)]


def make_data_subset(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    ratio = 15,
    subset_name = 'trial',
    exclude_dataset = 'val_2'
):

    # Generate folder path strings
    image_folder = path_gen([
        path,
        'data',
        data_set,
        'images'
    ])
    annot_folder = path_gen([
        path,
        'data',
        data_set,
        'annotations'
    ])

    filenames = os.listdir(f'{image_folder}all/')
    image_names = [filename for filename in filenames if filename.endswith('.tif')]
    image_names.sort()
    dataset_names = [image_name for iI, image_name in enumerate(image_names) if iI % ratio == 0]
    extra_names = [image_name for iI, image_name in enumerate(image_names) if ((iI+1)%len(image_names)) % ratio == 0 and (image_name.startswith('A172') or image_name.startswith('BV2') or image_name.startswith('SHSY5Y'))]
    dataset_names.extend(extra_names)
    dataset_names = sorted(list(set(dataset_names)))

    if exclude_dataset:
        exclude_filenames = os.listdir(f'{image_folder}{exclude_dataset}/')
        dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name not in exclude_filenames]

    # Create output directories if missing
    if not os.path.isdir(f'{image_folder}{subset_name}'):
        os.makedirs(f'{image_folder}{subset_name}')
    if not os.path.isdir(f'{annot_folder}{subset_name}'):
        os.makedirs(f'{annot_folder}{subset_name}')

    for filename in dataset_names:
        shutil.copyfile(
            f'{image_folder}all/{filename}', 
            f'{image_folder}{subset_name}/{filename}'
        )
        shutil.copyfile(
            f'{annot_folder}all/{filename}', 
            f'{annot_folder}{subset_name}/{filename}'
        )


def remove_bad_annotations(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    subset = 'val_2',
):

    # Generate folder path strings
    image_folder = path_gen([
        path,
        'data',
        data_set,
        'images'
    ])
    annot_folder = path_gen([
        path,
        'data',
        data_set,
        'annotations'
    ])

    print('\tRemoving one cell and weird files from both the image and annotation folder...')

    one_cell_annotation_files = []
    with open(f'{annot_folder}one_cell_annotations.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            one_cell_annotation_files.append(f'{args[0]}.tif')

    weird_annotation_files = []
    with open(f'{annot_folder}weird_annotations.txt', 'r') as infile:
        for line in infile:
            args = line.split('\n')
            weird_annotation_files.append(f'{args[0]}.tif')
    
    for file in one_cell_annotation_files:
        try:
            os.remove(f'{image_folder}{subset}/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}{subset}/{file}')
        except OSError:
            pass

    for file in weird_annotation_files:
        try:
            os.remove(f'{image_folder}{subset}/{file}')
        except OSError:
            pass
        try:
            os.remove(f'{annot_folder}{subset}/{file}')
        except OSError:
            pass

    print('\tDone')


def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values


def gpu_every_sec():
    """
        This function calls itself every 5 secs and print the gpu_memory.
    """
    Timer(1.0, gpu_every_sec).start()
    print(get_gpu_memory())


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