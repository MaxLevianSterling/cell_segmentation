import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import numpy                    as np
from torch.autograd             import Variable
from FusionNet                  import * 
from builders                   import Builder
from loaders                    import Loader
from image_transforms           import Compose
from image_transforms           import ToTensor
from image_transforms           import ToUnitInterval
from image_transforms           import FullCrop
from image_transforms           import StackOrient
from image_transforms           import ToNormal
from image_transforms           import Squeeze
from image_transforms           import ToBinary
from image_transforms           import StackReorient
from image_transforms           import StackMean
from image_transforms           import Uncrop
from utils                      import path_gen
from utils                      import get_gpu_list
from inspect                    import getargspec
from math                       import ceil


def deploy(

    # Data
    path                = '/mnt/sdg/maxs',
    annot_type          = 'soma', 

    # GPU
    n_gpus              = 4,
    kill_my_gpus        = False,
    reserved_gpus       = [6, 7],
    gpu_check_duration  = 5,
    gpu_usage_limit     = 500,

    # Loading
    load_data_set       = 'LIVECell',
    load_data_type      = 'part_set',
    load_data_subset    = '5',    
    load_subset_type    = 'train',
    load_model_name     = 'model_name',
    load_chkpt          = 1000,

    # Testing
    test_data_set       = 'LIVECell',
    test_data_type      = 'part_set',
    test_data_subset    = '5',
    test_subset_type    = 'test',
    test_dataset_device = 'cpu',

    # Model
    in_chan             = 1,
    out_chan            = 1,
    ngf                 = 64,
    spat_drop_p         = .5,
    act_fn_encode       = nn.LeakyReLU(negative_slope=.1),
    act_fn_decode       = nn.LeakyReLU(negative_slope=.1),
    act_fn_output       = nn.LeakyReLU(negative_slope=.1),
    init_name           = 'XU',
    init_gain           = 'leaky_relu',
    init_param          = .1,
    fan_mode            = 'fan_out',
    act_a_trans         = True, 
    bn_a_trans          = True, 
    bn                  = True,
    act_after_bn        = False,
    bn_momentum         = .1,

    # Transform
    crop_size           = 256,
    orig_size           = (520, 704),
    overlap             = 3,
    new_mean            = 0.,
    new_std             = 1.,
    tobinary            = .75,

    # Loader
    batch_size          = 32,
    shuffle             = False,
    drop_last           = False,

    # Output
    print_sep           = '$',
    n_images_saved      = 10,
):  
    """Deploy a cell segmentation network
    
    Args:
        Data
        path (string): base path for all folders
        annot_type (string): annotation style (soma, thin
            membrane, thick membrane)

        GPU
        n_gpus (int): number of GPUs needed
        kill_my_gpus (bool): whether to kill GPUs 2 and 3
        reserved_gpus (list): list of reserved device IDs
        gpu_check_duration (int): GPU observation time in 
            seconds
        gpu_usage_limit (int): free memory required on each 
            GPU in MB

        Loading
        load_data_set (string): to be loaded data set ( i.e.
            LIVECell)
        load_data_type (string): to be loaded data type (
            i.e. per_celltype, part_set, full_set)
        load_data_subset (string): to be loaded data subset (
            e.g. BV2, 50%)
        load_subset_type (string): to be loaded data subset type
            (i.e. train, test, val)
        load_model_name (string): to be loaded model identifier
        load_chkpt (int): training epoch age of saved
            checkpoint to start training at (0 -> no checkpoint 
            loading)

        Testing
        test_data_set (string): testing data set ( i.e.
            LIVECell)
        test_data_type (string): testing data type (
            i.e. per_celltype, part_set, full_set)
        test_data_subset (string): testing data subset (
            e.g. BV2, 50%)
        test_subset_type (string): testing data subset 
            type (i.e. train, test, val)
        test_dataset_device (string): what device to handle 
            testing data on

        Model
        in_chan (int): input channel number 
        out_chan (int): output channel number 
        ngf (int): channel depth factor 
        spat_drop_p (float): spatial dropout probability
        act_fn_encode (nn.Module): encoding activation function
        act_fn_decode (nn.Module): decoding activation function
        act_fn_output (nn.Module): output activation function  

        Transform
        crop_size (int / tuple of ints): HxW output dimensions
            of crops 
        orig_size (int / tuple of ints): HxW input dimensions
            of original data 
        overlap (int/float): maximum boosting overlap of cropped
            predictions
        new_mean (float): normalized mean of the input images
        new_std (float): normalized standard deviation of the 
            input images
        tobinary (float): float to binary cutoff point 

        Loader
        batch_size (int): data batch size
        shuffle (bool): whether input data is to be shuffled
        drop_last (bool): whether to use the last unequally 
            sized batch per dataset consumption

        Output
        print_sep (string): print output separation
            character
        n_images_saved (int): number of images saved every 
            save_image_interval
                
    Returns:
        Boosted binary cell segmentation predictions in 
            the results subfolder
    """
    
    # Being beautiful is not a crime
    print('\n', f'{print_sep}' * 87, '\n', sep='')

    # Generate folder path strings
    load_model_folder = path_gen([
        path,
        'models',
        load_data_set,
        load_data_type,
        load_data_subset,  
        load_subset_type,
        annot_type,
        load_model_name,
    ])
    results_folder = path_gen([
        path,
        'results',
        test_data_set,
        test_data_type,
        test_data_subset,
        test_subset_type,
        annot_type,
        load_model_name,
        'deployment'
    ])

    # Create output directories
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)  
        
    # Save deployment parameters
    deployment_variable_names = getargspec(deploy)[0]
    deployment_parameters = getargspec(deploy)[3] 
    with open(f'{load_model_folder}deployment_parameters.txt', 'w') as outfile:
        for iVn, iP in zip(deployment_variable_names, deployment_parameters):
            args = f'{iVn},{iP}\n'
            outfile.write(args)
            
    # Get list of GPUs to use
    gpu_device_ids = [0,1,2,3]
    # get_gpu_list(
    #     n_gpus,
    #     kill_my_gpus,
    #     reserved_gpus,
    #     gpu_check_duration,
    #     gpu_usage_limit,
    # )

    # Assign devices 
    if torch.cuda.is_available(): 
        nn_handler_device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\n\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to deploy model')
    
    # Indicate how output will be saved
    print(f'\tBoosted binary cell segmentation predictions will be saved in path/results...')
    print('\n\t', f'{print_sep}' * 71, '\n', sep='')   

    # Disable gradient calculation
    with torch.no_grad():

        # Get testing dataset and loader
        test_dset = Builder(
            path=path,
            data_set=test_data_set,
            data_type=test_data_type,
            data_subset=test_data_subset,
            subset_type=test_subset_type,
            annot_type=annot_type,
            dataset_device=test_dataset_device,
            deploy=True,
            supervised_offline_transforms=Compose([
                ToTensor(type=torch.float),
                ToUnitInterval(items=[0, 1]),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std),
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
        test_loader = Loader(
            test_dset,
            dataset_device=test_dataset_device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        
        # Initiate model
        Model = nn.DataParallel(
            FusionGenerator(
                in_chan, 
                out_chan, 
                ngf,
                spat_drop_p,
                act_fn_encode,
                act_fn_decode,
                act_fn_output,
                init_name,
                init_gain,
                init_param,
                fan_mode,
                act_a_trans, 
                bn_a_trans, 
                bn,
                act_after_bn,
                bn_momentum,
            ).to(device=nn_handler_device), 
            device_ids=gpu_device_ids,
            output_device=nn_handler_device,
        )

        # Load trained network
        chkpt_path = f'{load_model_folder}checkpoint{load_chkpt}.tar'
        chkpt = torch.load(chkpt_path, map_location=nn_handler_device)
        load_check = Model.module.load_state_dict(chkpt['model_module_state_dict'])
        if not load_check:
            raise RuntimeError('\n\tNot all module parameters loaded correctly')
        print(f'\tCheckpoint of model {load_model_name} at epoch {load_chkpt} restored...') 
        print('\n\t', f'{print_sep}' * 71, '\n', sep='')

        # Make sure evaluation mode is enabled
        Model.eval()
    
        # Initialize local variables
        n_crops_h = ceil(overlap * (orig_size[0] - crop_size) / crop_size) + 1
        n_crops_w = ceil(overlap * (orig_size[1] - crop_size) / crop_size) + 1
        n_crops = n_crops_h * n_crops_w 
        predictions = [
            torch.zeros(
                (len(test_dset), 8, crop_size, crop_size), 
                dtype=torch.float, 
                device=test_dataset_device
            )
            for iC in range(n_crops)
        ]

        # Perform online epoch image transformations
        Builder.epoch_pretransform(test_dset)

        # Deploy model
        for iB, batch in enumerate(test_loader):
            
            # Initialize batch variables
            current_batch_len = list(batch['image'][0].size())[0]

            # Feed image set through model
            for iC in range(n_crops):
                for iO in range(8):
                    with torch.cuda.amp.autocast():
                        x = Variable(batch['image'][iC][:,iO:iO+1,:,:]).to(device=nn_handler_device)
                        y = Model(x)
   
                    # v_utils.save_image(
                    #     x[0].detach().to('cpu').type(torch.float32),
                    #     f'{results_folder}image_checkpoint{load_chkpt}.png'
                    # )
                    # v_utils.save_image(
                    #     y[0].detach().to('cpu').type(torch.float32),
                    #     f'{results_folder}pred_checkpoint{load_chkpt}.png'
                    # )

                    predictions[iC][
                        iB*batch_size : iB*batch_size+current_batch_len,
                        iO:iO+1,
                        :,
                        :
                    ] = y.detach().to('cpu')
            
                # Display progress
                batch_ratio = (iB + 1) / (len(test_loader))
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
        predictions = Builder.epoch_posttransform(
            test_dset, 
            {'image': predictions}
        )

        # Save predictions in results folder
        np.save(
            f'{results_folder}checkpoint{load_chkpt}_predictions.npy', 
            predictions['image'].detach().to('cpu').numpy()
        )

        # Save example network image outputs
        image_idxs = torch.randperm(n_images_saved)
        for idx in range(n_images_saved):
            iI = image_idxs[idx].item()
            v_utils.save_image(
                predictions['image'][iI, :, :].detach().to('cpu').type(torch.float32),
                f'{results_folder}pred_checkpoint{load_chkpt}_image{iI}.png'
            )


# Run deploy() if deploy.py is run directly
if __name__ == '__main__':

    # Run train()
    deploy()
    print('\n\n', end='')