from utils import *
#import argparse
#import os


parser = argparse.ArgumentParser(description='Preprocesses data if necessary')
parser.add_argument('-p', 
                    '--path',
                    type=str,
                    nargs='?',
                    default='C:/Users/Max-S/tndrg/Data/LIVECell/',
                    help='The directory path to the data')
parser.add_argument('-s',
                    '--subset',
                    type=str,
                    nargs='?',
                    default='trial',
                    help='The data subset to preprocess')
args = parser.parse_args()

"""Preprocesses LIVECell image and annotation data

Note:
    Required file structure:
        <path>/
            images/
                <subset>/<.tif files>
            annotations/
                <subset>/<.tif files>
                (<.json file>)
    Make sure that the image subset folder is complete

Args:
    path (string): input directory path
    subset (string): data subset directory

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

print(f'\nPreprocessing "{args.subset}" data in folder "{args.path}"')

#########################################################################################################
# Image data
print('\nChecking for image data... ')
img_folder = f'{args.path}images/{args.subset}/'
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

print('Checking for preprocessed image variables... ')
img_var_folder_exists = os.path.isdir(f'{img_folder}variables/')
img_var_filenames_exists = os.path.isfile(f'{img_folder}variables/filenames.txt')
img_var_tifstack_exists = os.path.isfile(f'{img_folder}variables/stack.tif')
img_var_arr_exists = os.path.isfile(f'{img_folder}variables/array.npy')
img_var_oriented_arr_exists = os.path.isfile(f'{img_folder}variables/oriented_array.npy')
img_var_oriented_filenames_exists = os.path.isfile(f'{img_folder}variables/oriented_filenames.txt')

if not img_var_folder_exists:
    print('No preprocessed variables found...')

if img_var_filenames_exists and \
        img_var_tifstack_exists and \
        img_var_arr_exists and \
        img_var_oriented_arr_exists and \
        img_var_oriented_filenames_exists:
    print('All preprocessed variables found...')
else:
    if not img_var_filenames_exists or not img_var_tifstack_exists:
        print('\tMissing .tif stack or filenames. Creating now...')
        stack_tifs(img_folder)

    if not img_var_arr_exists:
        print('\tMissing array. Creating now...')
        tif_stack2arr(img_folder)

    if not img_var_oriented_arr_exists or not img_var_oriented_filenames_exists:
        print('\tMissing oriented array or oriented filenames. Creating now...')
        stack_orient(img_folder)

#########################################################################################################
# Annotation data
ann_folder = f'{args.path}annotations/{args.subset}/'
ann_folder_exists = os.path.isdir(ann_folder)
n_tifs_anns = 0
if ann_folder_exists:
    for file in os.listdir(ann_folder):
        if file.endswith('.tif'):
            n_tifs_anns += 1

print('\nChecking for preprocessed annotation variables... ')
ann_var_folder_exists = os.path.isdir(f'{ann_folder}variables/')
ann_var_filenames_exists = os.path.isfile(f'{ann_folder}variables/filenames.txt')
ann_var_tifstack_exists = os.path.isfile(f'{ann_folder}variables/stack.tif')
ann_var_arr_exists = os.path.isfile(f'{ann_folder}variables/array.npy')
ann_var_oriented_arr_exists = os.path.isfile(f'{ann_folder}variables/oriented_array.npy')
ann_var_oriented_filenames_exists = os.path.isfile(f'{ann_folder}variables/oriented_filenames.txt')

if not ann_var_folder_exists:
    print('No preprocessed variables found...')

if ann_var_filenames_exists and \
        ann_var_tifstack_exists and \
        ann_var_arr_exists and \
        ann_var_oriented_arr_exists and \
        ann_var_oriented_filenames_exists:
    print('All preprocessed variables found...')
else:
    print('\nChecking for annotation data...')
    if not ann_folder_exists or n_tifs_anns != n_tifs_imgs and not ann_var_folder_exists:
        print('\tMissing .tif files. Checking for .json file... ', end='')
        json_file_exists = os.path.isfile(f'{args.path}annotations/{args.subset}.json')
        if json_file_exists:
            print(f'Found. Processing for ~{round(n_tifs_imgs*3.5/60, 1)} minutes... ')      
            with HiddenPrints():
                json2array(f'{args.path}annotations/{args.subset}.json', create_tifs=True)
        else:
            raise RuntimeError('Annotation data not found.\n')

    if not ann_var_filenames_exists or not ann_var_tifstack_exists:
        print('\tMissing .tif stack or filenames. Creating now...')
        stack_tifs(ann_folder)

    if not ann_var_arr_exists:
        print('\tMissing array. Creating now...')
        tif_stack2arr(ann_folder)

    if not ann_var_oriented_arr_exists or not ann_var_oriented_filenames_exists:
        print('\tMissing oriented array or oriented filenames. Creating now...')
        stack_orient(ann_folder)