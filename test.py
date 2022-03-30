import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell
from image_transforms           import Compose
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import ToNormal
from utils                      import path_gen
from utils                      import worker_init_fn
from math                       import ceil, floor
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec


def test(

    # Data variables
    path                = '/mnt/sdg/maxs',
    data_set            = 'LIVECell',
    data_subset         = 'trial',
    model_data_set      = 'LIVECell',
    model_data_subset   = 'trial',

    # GPU variables
    gpu_device_ids      = 'all_available',
    reserved_gpus       = [0, 1, 6, 7],

    # Model variables
    model               = 'base_128',
    load_chkpt          = 2000,
    save_images         = 5,

    # Transform variables
    crop_size           = 128,
    orig_size           = (520, 704),
    new_mean            = .5,
    new_std             = .15,

    # DataLoader variables
    batch_size          = 200,
    shuffle             = True,
    num_workers         = 0,
    pin_memory          = True,
    persistent_workers  = False,
    prefetch_factor     = 2,
    drop_last           = False,

    # Verbosity variables
    print_sep           = '$',
):  
    """Tests a FusionNet-type neural network quickly,
        without boosting or full deployment, just to 
        gauge network performance in terms of loss.
    
    Note:
        Image data must be grayscale
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            <data_subset>/<.tif files>
                        annotations/
                            <data_subset>/<.tif files>
                            (<.json file>)
                models/
                results/

    Args:
        path (string): path to test data folder
        data_set (string): test data set 
        data_subset (string): test data subset 
        model_data_set (string): trained model data set 
        model_data_subset (string): trained model data subset 
        
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        reserved_gpus (list of ints): reserved GPUs not to be
            used
            
        model (string): current model identifier
        load_chkpt (int): training epoch age of saved 
            checkpoint to start testing at
        save_images (int): amount of test output images to save
            (max = batch_size)
        
        crop_size (int / tuple of ints): HxW output dimensions
            of crops 
        orig_size (int / tuple of ints): HxW input dimensions
            of original data 
        new_mean (float): normalized mean of the input images
        new_std (float): normalized standard deviation of the 
            input images
        
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
        A log of all testing set batch losses in the results 
            subfolder 
        (Optional) Example network image outputs in the
            results subfolder
        
    Raises:
        RuntimeError: At least one GPU must be available to 
            test FusionNet
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
        'testing'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Save testing parameters
    testing_variable_names = getargspec(test)[0]
    testing_parameters = getargspec(test)[3] 
    with open(f'{models_folder}FusionNet_testing_parameters.txt', 'a') as outfile:
        for iVn, iP in zip(testing_variable_names, testing_parameters):
            args = f'{iVn},{iP}\n'
            outfile.write(args)
    
    # Get available GPUs
    if gpu_device_ids == 'all_available':
        gpu_device_ids = getAvailable(
            limit=100, 
            maxLoad=0.05, 
            maxMemory=0.05
        )

    # Remove reserved GPUs
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
        raise RuntimeError('\n\tAt least one GPU must be available to train FusionNet')

    # Indicate how output will be saved
    print(f'\tAfter completion, {save_images} example outputs will be saved in path/results...')
    print('\n\t', f'{print_sep}' * 71, '\n', sep='')
    
    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_test_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            dataset_device=dataset_device,
            offline_transforms=Compose([
                ToUnitInterval(),
                ToTensor(),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std)
            ]),
            epoch_pretransforms=Compose([
                RandomCrop(input_size=orig_size, output_size=crop_size),
                RandomOrientation(),
            ])
        )

        # Reset number of workers if necessary
        if num_workers == 'ratio':
            num_workers = len(LIVECell_test_dset) / batch_size
            if isinstance(num_workers, float) and not drop_last:
                num_workers = ceil(num_workers)
            else:
                num_workers = floor(num_workers)

        # Initiate standard DataLoader() instance
        dataloader = data.DataLoader(
            LIVECell_test_dset, 
            batch_size=batch_size,
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

        # Define loss function
        loss_func = nn.SmoothL1Loss()

        # Initiate FusionNet
        FusionNet = nn.DataParallel(
            FusionGenerator(1,1,64).to(device=nn_handler_device), 
            device_ids=gpu_device_ids,
            output_device=nn_handler_device
        )
    
        # Load trained network
        chkpt_path = f'{models_folder}FusionNet_checkpoint{load_chkpt}.tar'
        chkpt = torch.load(chkpt_path, map_location=nn_handler_device)
        load_check = FusionNet.module.load_state_dict(chkpt['model_module_state_dict'])
        if not load_check:
            raise RuntimeError('\n\tNot all module parameters loaded correctly')
        print(f'\tTesting with checkpoint of epoch {load_chkpt} of model {model} trained on {model_data_set}/{model_data_subset}...')
        print(f'\tUsing network to test on images from {data_set}/{data_subset}...')   
        print('\n\t', f'{print_sep}' * 71, '\n', sep='')
       
        # Make sure evaluation mode is enabled
        FusionNet.eval()

        # Initialize loss log
        batch_loss_log = []
        
        # Perform online epoch image transformations
        LIVECell.epoch_transform(LIVECell_test_dset)

        # Test FusionNet
        for iB, batch in enumerate(dataloader):
            
            # Wrap the batch and pass it forward
            x = Variable(batch['image']).to(device=nn_handler_device)
            y_ = Variable(batch['annot']).to(device=nn_handler_device)
            y = FusionNet(x)
            loss = loss_func(y, y_)

            # Calculate and record loss
            batch_loss_log.append(loss.item())

            # Display progress
            batch_ratio = (iB) / (len(dataloader) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\Images: [{:<{}}] {:.0f}%; Loss: {:.5f}".format(
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()

        # Display average loss
        print(f'\n\tAverage testing loss: {mean(batch_loss_log)}')

        # Loss log output
        with open(f'{results_folder}FusionNet_test_loss_from_checkpoint{load_chkpt}.txt', 'w') as outfile:
            for batch_loss in batch_loss_log:
                args = f'{batch_loss}\n'
                outfile.write(args)
        
        # Optional example network image outputs
        if save_images:
            
            # Clip amount of images to be saved if necessary
            last_batch_size = list(batch.values())[0].shape[0]
            if save_images > last_batch_size:
                save_images = last_batch_size
                print(f'\tAmount of images to be saved has been reset to maximum {save_images}...')

            for iI in torch.randperm(save_images):
                iI = iI.item()
                v_utils.save_image(
                    x[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_image_checkpoint{load_chkpt}_image{iI}.png'
                )
                v_utils.save_image(
                    y_[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_annot_checkpoint{load_chkpt}_image{iI}.png'
                )
                v_utils.save_image(
                    y[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_pred_checkpoint{load_chkpt}_image{iI}.png'
                )


# If test.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3 that take up memory
    os.system("""kill $(nvidia-smi | awk '$5=="PID" && $8>0 {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run test()
    test()
    print('\n\n', end='')