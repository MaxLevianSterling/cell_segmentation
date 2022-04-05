import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import numpy                    as np
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell
from dataloaders                import GPU_dataloader 
from image_transforms           import Compose
from image_transforms           import FullCrop
from image_transforms           import StackOrient
from image_transforms           import ToNormal
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from image_transforms           import Squeeze
from image_transforms           import ToBinary
from image_transforms           import StackReorient
from image_transforms           import StackMean
from image_transforms           import Uncrop
from utils                      import path_gen
from utils                      import get_gpu_memory
from utils                      import gpu_every_sec
from math                       import ceil
from GPUtil                     import getAvailable
from inspect                    import getargspec


def deploy(

    # Data variables
    path                = '/mnt/sdg/maxs',
    data_set            = 'LIVECell',
    data_subset         = 'val_2',
    model_data_set      = 'LIVECell',
    model_data_subset   = 'trial',

    # GPU variables
    gpu_device_ids      = 'all_available',
    reserved_gpus       = [4, 6, 7],

    # Training variables
    load_chkpt          = 20000,
    amp                 = True,

    # Model variables
    model               = 'base_128',
    in_chan             = 1,
    out_chan            = 1,
    ngf                 = 64,
    spat_drop_p         = .05,
    act_fn_encode       = nn.LeakyReLU(0.2),
    act_fn_decode       = nn.ReLU(),
    act_fn_output       = nn.Tanh(),

    # Transform variables
    crop_size           = 128,
    orig_size           = (520, 704),
    overlap             = 3,
    new_mean            = .5,
    new_std             = .15,
    tobinary            = .3,

    # DataLoader variables
    batch_size          = 128,
    shuffle             = False,
    drop_last           = False,

    # Verbosity variables
    print_sep           = '$',
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
        
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        reserved_gpus (list of ints): reserved GPUs not to be
            used
            
        model (string): current model identifier
        load_chkpt (int): training epoch age of saved 
            checkpoint to start deployment at
        amp (bool): whether automatic precision is to be used
        
        crop_size (int / tuple of ints): HxW output dimensions
            of crops 
        orig_size (int / tuple of ints): HxW input dimensions
            of original data 
        new_mean (float): normalized mean of the input images
        new_std (float): normalized standard deviation of the 
            input images
        tobinary (float): float to binary cutoff point 
        
        batch_size (int): data batch size (default = <maximum>)
        shuffle (bool): whether input data is to be shuffled
        num_workers (int): number of workers to be used for 
            multi-process data loading (default = 'ratio' ->
            spawns a worker for each batch in an epoch)
        pin_memory (bool): tensors fetched by DataLoader pinned
            in memory, enabling faster data transfer to
            CUDA-enabled GPUs.
        persistent_workers (bool): worker processes will not be
            shut down after a dataset has been consumed once,
            allowing the workers Dataset instances to remain
            alive. 
        prefetch_factor (int): number of samples to prefetch by
            multi-process workers
        drop_last (bool): whether to use the last unequally 
            sized batch per dataset consumption
            
        print_sep (string): print output separation
            character
                
    Returns:
        Boosted binary cell segmentation predictions in 
            the results subfolder
    """
    
    # Being beautiful is not a crime
    print('\n', f'{print_sep}' * 87, '\n', sep='')

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

    # Assign devices 
    dataset_device = 'cpu' 
    if torch.cuda.is_available(): 
        nn_handler_device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[-1]} as online dataset storage...')
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to deploy FusionNet')
    
    # Indicate how output will be saved
    print(f'\tBoosted binary cell segmentation predictions will be saved in path/results...')
    print('\n\t', f'{print_sep}' * 71, '\n', sep='')   

    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_deploy_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            dataset_device=dataset_device,
            deploy=True,
            offline_transforms=Compose([
                ToUnitInterval(),
                ToTensor(),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std)
            ]),
            epoch_pretransforms=Compose([
                FullCrop(input_size=orig_size, output_size=crop_size, overlap=overlap),
                StackOrient()
            ]),
            epoch_posttransforms=Compose([
                StackReorient(),
                StackMean(),
                Uncrop(input_size=crop_size, output_size=orig_size, overlap=overlap),
                ToBinary(cutoff=tobinary, items=[0]),
                Squeeze(),
            ])
        )

        #Initiate custom data loader
        dataloader = GPU_dataloader(
            LIVECell_deploy_dset,
            dataset_device=dataset_device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        
        # Initiate FusionNet
        FusionNet = nn.DataParallel(
            FusionGenerator(
                in_chan, 
                out_chan, 
                ngf,
                spat_drop_p,
                act_fn_encode,
                act_fn_decode,
                act_fn_output
            ).to(device=nn_handler_device), 
            device_ids=gpu_device_ids,
            output_device=nn_handler_device,
        )

        # Load trained network
        chkpt_path = f'{models_folder}FusionNet_checkpoint{load_chkpt}.tar'
        chkpt = torch.load(chkpt_path, map_location=nn_handler_device)
        load_check = FusionNet.module.load_state_dict(chkpt['model_module_state_dict'])
        if not load_check:
            raise RuntimeError('\n\tNot all module parameters loaded correctly')
        print(f'\tDeploying checkpoint of epoch {load_chkpt} of model {model} trained on {model_data_set}/{model_data_subset}...')
        print(f'\tUsing network to segment cells in images from {data_set}/{data_subset}...')       
        print('\n\t', f'{print_sep}' * 71, '\n', sep='')

        # Make sure evaluation mode is enabled
        FusionNet.eval()
    
        # Initialize local variables
        n_crops_h = ceil(
            overlap * (orig_size[0] - crop_size
        ) / crop_size) + 1
        n_crops_w = ceil(
            overlap * (orig_size[1] - crop_size
        ) / crop_size) + 1
        n_crops = n_crops_h * n_crops_w 
        predictions = [
            torch.zeros(
                (len(LIVECell_deploy_dset), 8, crop_size, crop_size), 
                dtype=torch.float, 
                device=dataset_device
            )
            for iC in range(n_crops)
        ]

        # Perform online epoch image transformations
        LIVECell.epoch_pretransform(LIVECell_deploy_dset)

        # Deploy FusionNet
        for iB, batch in enumerate(dataloader):
            
            # Initialize batch variables
            current_batch_len = list(batch['image'][0].size())[0]

            # Feed image set through FusionNet
            for iC in range(n_crops):
                for iO in range(8):
                    if amp:
                        with torch.cuda.amp.autocast():
                            x = Variable(batch['image'][iC][:,iO:iO+1,:,:]).to(device=nn_handler_device)
                            y = FusionNet(x)
                    else:
                        x = Variable(batch['image'][iC][:,iO:iO+1,:,:]).to(device=nn_handler_device)
                        y = FusionNet(x)
   
                    # v_utils.save_image(
                    #     x[0].detach().to('cpu').type(torch.float32),
                    #     f'{results_folder}FusionNet_image_checkpoint{load_chkpt}.png'
                    # )
                    # v_utils.save_image(
                    #     y[0].detach().to('cpu').type(torch.float32),
                    #     f'{results_folder}FusionNet_pred_checkpoint{load_chkpt}.png'
                    # )

                    predictions[iC][
                        iB*batch_size : iB*batch_size+current_batch_len,
                        iO:iO+1,
                        :,
                        :
                    ] = y.detach().to('cpu')
            
                # Display progress
                batch_ratio = (iB + 1) / (len(dataloader))
                crop_ratio = (iC + 1) / (n_crops)
                sys.stdout.write('\r')
                sys.stdout.write(
                    "\tBatches: [{:<{}}] {:.0f}%; Crops: [{:<{}}] {:.0f}%    ".format(
                        "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                        "=" * int(20*crop_ratio), 20, 100*crop_ratio,
                    )
                )
                sys.stdout.flush()
                
        # Prediction boosting with uncropping
        predictions = LIVECell.epoch_posttransform(
            LIVECell_deploy_dset, 
            {'image': predictions}
        )

        # Save predictions in results folder
        np.save(
            f'{results_folder}FusionNet_checkpoint{load_chkpt}_predictions.npy', 
            predictions['image']
        )


# Run deploy() if deploy.py is run directly
if __name__ == '__main__':

    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    deploy()
    print('\n\n', end='')