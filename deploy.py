import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import torchvision.transforms   as transforms
import numpy                    as np
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell_trial
from image_transforms           import FullCrop
from image_transforms           import StackOrient
from image_transforms           import ToNormal
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
from inspect                    import getargspec
from math                       import ceil


def deploy(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'val_2',
    model_data_set = 'LIVECell',
    model_data_subset = 'trial',
    print_separator = '$',
    gpu_device_ids = 'all_available',
    model = 'ReLuX_128_deform',
    load_snapshot = 1000,
    pred_size = 128,
    orig_size = (520, 704),
    new_mean = .5,
    new_std = .15,
    tobinary = .9,
    batch_size = 16,
    num_workers = 0,
    prefetch_factor = 2,
    pin_memory = True,
    persistent_workers = False,
    amp = False,
    reserved_gpus = [6, 7]
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
            character
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier 
        load_snapshot (int): training epoch age of saved 
            snapshot to start deployment at
        tobinary (float): float to binary cutoff point
                
    Returns:
        Boosted binary cell segmentation predictions in 
            the results subfolder
    """
    
    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 87, '\n', sep='')

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
        
    # Save deployment parameters
    deployment_variable_names = getargspec(deploy)[0]
    deployment_parameters = getargspec(deploy)[3] 
    with open(f'{models_folder}FusionNet_training_parameters.txt', 'a') as outfile:
        for iVn, iP in zip(deployment_variable_names, deployment_parameters):
            args = f'{iVn},{iP}\n'
            outfile.write(args)
            
    # Get available GPUs
    if gpu_device_ids == 'all_available':
        gpu_device_ids = getAvailable(
            limit=100, 
            maxLoad=0.05, 
            maxMemory=0.05
        )
    
    # Remove reserved GPUs from available list
    gpu_device_ids = [
        gpu 
        for gpu in gpu_device_ids 
        if gpu not in reserved_gpus
    ]

    # Assign devices if CUDA is available
    if torch.cuda.is_available(): 
        dataset_device = f'cuda:{gpu_device_ids[-1]}' 
        nn_handler_device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[-1]} as online dataset storage...')
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to deploy FusionNet')
    
    # Indicate how output will be saved
    print(f'\tBoosted binary cell segmentation predictions will be saved in path/results...')
    print('\n\t', f'{print_separator}' * 71, '\n', sep='')   

    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_deploy_dset = LIVECell_trial(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            dataset_device=dataset_device,
            deploy=True,
            offline_transform=transforms.Compose([
                ToUnitInterval(),
                ToTensor(),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std)
            ]),
            online_epoch_pretransform=transforms.Compose([
                FullCrop(input_size=orig_size, output_size=pred_size),
                StackOrient()
            ]),
            online_epoch_posttransform=transforms.Compose([
                StackReorient(),
                StackMean(),
                Uncrop(input_size=pred_size, output_size=orig_size),
                Squeeze(),
                ToBinary(cutoff=tobinary, items=[0])
            ])
        )

        # Reset number of workers if necessary
        if num_workers == 'max':
            num_workers = len(gpu_device_ids)
        elif num_workers == 'ratio':
            num_workers = min([
                ceil(len(LIVECell_deploy_dset)/batch_size), 
                len(gpu_device_ids)
            ])

        # Initiate standard DataLoader() instance
        dataloader = data.DataLoader(
            LIVECell_deploy_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        # Initiate FusionNet
        FusionNet = FusionGenerator(1,1,64).to(device=nn_handler_device, dtype=torch.float)
    
        # Load trained network
        model_path = f'{models_folder}FusionNet_snapshot{load_snapshot}.tar'
        checkpoint = torch.load(model_path)
        FusionNet.load_state_dict(checkpoint['model_module_state_dict'])
        print(f'\tDeploying snapshot of epoch {load_snapshot} of model {model} trained on {model_data_set}/{model_data_subset}...')
        print(f'\tUsing network to segment cells in images from {data_set}/{data_subset}...')       
        print('\n\t', f'{print_separator}' * 71, '\n', sep='')

        # Wrap model for parallel GPU usage
        FusionNet = nn.DataParallel(
            FusionNet, 
            device_ids=gpu_device_ids
        )

        # Make sure evaluation mode is enabled
        FusionNet.eval()
 
        # Initialize batch loss log
        batch_loss_log = []

        # Perform online epoch image transformations
        LIVECell_trial.epoch_transform(LIVECell_deploy_dset)

        # Deploy FusionNet
        for iB, batch in enumerate(dataloader):
            
            # Initialize local variables
            predictions = []
            current_batch_size = list(batch['image'][0].size())[0]
        
            # Feed image set through FusionNet
            for iC in range(len(batch['image'])):
                predictions.append(
                    torch.zeros(
                        (len(LIVECell_deploy_dset), 8, pred_size, pred_size), 
                        dtype=torch.float, 
                        device=nn_handler_device
                    )
                )
                for iO in range(8):
                    if amp:
                        with torch.cuda.amp.autocast():
                            x = Variable(batch['image'][iC][:,iO:iO+1,:,:]).to(device=nn_handler_device, dtype=torch.float)
                            y = FusionNet(x)
                    else:
                        x = Variable(batch['image'][iC][:,iO:iO+1,:,:]).to(device=nn_handler_device, dtype=torch.float)
                        a = x[0]
                        v_utils.save_image(
                            x[0].detach().to('cpu').type(torch.float32),
                            f'{results_folder}FusionNet_image_snapshot{load_snapshot}.png'
                        )
                        y = FusionNet(x)
                        b = y[0]
                        v_utils.save_image(
                            y[0].detach().to('cpu').type(torch.float32),
                            f'{results_folder}FusionNet_pred_snapshot{load_snapshot}.png'
                        )
            
                    predictions[iC][
                        iB*batch_size : iB*batch_size+current_batch_size,
                        iO:iO+1,
                        :,
                        :
                    ] = y
            
            # Display progress
            batch_ratio = (iB) / (len(dataloader) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tImages: [{:<{}}] {:.0f}%".format(
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                )
            )
            sys.stdout.flush()
        
        # Prediction boosting with uncropping
        predictions = LIVECell_trial.epoch_transform(LIVECell_deploy_dset, predictions)

        # Save predictions in results folder
        predictions = np.stack(predictions.detach().to('cpu').numpy(), axis=0)
        np.save(f'{results_folder}FusionNet_snapshot{load_snapshot}_prediction_array.npy', predictions)


# Run deploy() if deploy.py is run directly
if __name__ == '__main__':

    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    deploy()
    print('\n\n', end='')