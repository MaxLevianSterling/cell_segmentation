import os
import sys
import torch
import torchvision.transforms   as transforms
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell
from image_transforms           import StackCrop
from image_transforms           import StackOrient
from image_transforms           import Padding
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from image_transforms           import Squeeze
from image_transforms           import ToBinary
from image_transforms           import Unpadding
from image_transforms           import StackReorient
from image_transforms           import StackMean
from image_transforms           import Uncrop
from utils                      import path_gen
from GPUtil                     import getAvailable
import numpy                    as np


def deploy(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'extra',
    model_data_set = 'LIVECell',
    model_data_subset = 'trial',
    print_separator = '$',
    gpu_device_ids = getAvailable(
        limit=100, 
        maxLoad=0.1, 
        maxMemory=0.1
    ),
    model = '1',
    load_snapshot = 2,
):  
    """Deploys a FusionNet-type neural network
    to generate predictions for a testing set
    
    Note:
        Image data must be grayscale
        Multi-orientation boosting is performed such that 
            cropped and reoriented predictions are 
            overlayed and averaged to generate the final
            prediction
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            <data_subset>/<.tif files>
                models/
                results/

    Args:
        path (string): path to deployment data folder
        data_set (string): deployment data set 
        data_subset (string): deployment data subset 
        model_data_set (string): trained model data set 
        model_data_subset (string): trained model data subset 
        print_separator (string): print output separation
            character (default = '$')
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier (default = 1)
        load_snapshot (int): training epoch age of saved 
            snapshot to start deployment at (default = 1)
                
    Returns:
        Boosted binary cell segmentation predictions in 
            the results subfolder
    """
    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 87, '\n', sep='')

    # Assign devices if CUDA is available
    if torch.cuda.is_available(): 
        device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to deploy FusionNet')
    
    # Indicate how output will be saved
    print(f'\tBoosted binary cell segmentation predictions will be saved in path/results...')
    print('\n\t', f'{print_separator}' * 71, '\n', sep='')   

    # Generate folder path strings
    models_folder = path_gen([
        path,
        'models',
        model_data_set,
        model_data_subset,
        model
    ])
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_subset,
        model,
        'deployment'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    
    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_deploy_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            deploy=True,
            transform=None
        )

        # Define prerocessing transforms
        pretransform=transforms.Compose([
            StackCrop(input_size=(520,704), output_size=512),
            StackOrient(),
            Padding(width=64),
            ToUnitInterval(),
            ToTensor()
        ]) 
        
        # Define postrocessing transforms
        posttransform=transforms.Compose([
            Squeeze(),
            Unpadding(width=64),
            StackReorient(),
            StackMean(),
            Uncrop(input_size=512, output_size=(520,704)),
            ToBinary(cutoff=.5, items=[0])
        ]) 
        
        # Initiate FusionNet
        FusionNet = nn.DataParallel(
            FusionGenerator(1,1,64), 
            device_ids=gpu_device_ids
        ).to(device=device, dtype=torch.float)

        # Load trained network and set to evaluate
        model_path = f'{models_folder}FusionNet_snapshot{load_snapshot}.pkl'
        FusionNet.load_state_dict(torch.load(model_path))
        FusionNet.eval()
        print(f'\tDeploying snapshot of epoch {load_snapshot} of model {model} trained on {model_data_set}/{model_data_subset}...')
        print(f'\tUsing network to segment cells in images from {data_set}/{data_subset}...')       
        print('\n\t', f'{print_separator}' * 71, '\n', sep='')

        # Deploy FusionNet
        predictions = {'images': []}
        for iI, image in enumerate(LIVECell_deploy_dset):
            
            # Preprocess image for boosting
            image_set = pretransform(image)

            # Feed image set through FusionNet
            prediction = {'image':[]}
            for iC in range(len(image_set['image'])):
                x = Variable(image_set['image'][iC]).to(device=device, dtype=torch.float)
                y = FusionNet(x)
                prediction['image'].append(y.detach().to('cpu').numpy())
            
            # Prediction boosting with uncropping
            predictions['images'].append(posttransform(prediction)['image'])

            # Display progress
            image_ratio = (iI) / (len(LIVECell_deploy_dset) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tImages: [{:<{}}] {:.0f}%".format(
                    "=" * int(20*image_ratio), 20, 100*image_ratio,
                )
            )
            sys.stdout.flush()

        # Save predictions in results folder
        predictions['images'] = np.stack(predictions['images'], axis=0)
        np.save(f'{results_folder}prediction_array.npy', predictions['images'])

# Run deploy() if deploy.py is run directly
if __name__ == '__main__':

    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    deploy()
    print('\n', end='')