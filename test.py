import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import torchvision.transforms   as transforms
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell_trial
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import ToNormal
from image_transforms           import Noise
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from utils                      import path_gen
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec


def test(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'val_2',
    model_data_set = 'LIVECell',
    model_data_subset = 'trial',
    print_separator = '$',
    gpu_device_ids = 'all_available',
    model = 'ReLuX_128_deform',
    load_snapshot = 1500,
    new_mean = .5,
    new_std = .15,
    pred_size = 128,
    orig_size = (520, 704),
    localdeform = [8, 8],
    tobinary = .9,
    noise = .05,
    batch_size = 128,
    pin_memory = True, 
    prefetch_factor = 2,
    persistent_workers = True,
    save_images = 5,
    reserved_gpus = [6, 7]
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
        print_separator (string): print output separation
            character
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier
        load_snapshot (int): training epoch age of saved 
            snapshot to test
        batch_size (int): data batch size (default = <maximum>)
        pin_memory (bool): tensors fetched by DataLoader pinned 
            in memory, enabling faster data transfer to 
            CUDA-enabled GPUs.
        persistent_workers (bool): worker processes will not be 
            shut down after a dataset has been consumed once, 
            allowing the workers Dataset instances to remain 
            alive.
        save_images (int): amount of test output images to save
            (max = batch_size)
        
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
    print('\n\t', f'{print_separator}' * 71, '\n', sep='')
    
    # Set number of workers equal to number of GPUs used
    num_workers = len(gpu_device_ids)

    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_test_dset = LIVECell_trial(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            dataset_device=dataset_device,
            offline_transform=transforms.Compose([
                ToUnitInterval(),
                ToTensor(),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std)
            ]),
            online_epoch_pretransform=transforms.Compose([
                RandomCrop(input_size=orig_size, output_size=pred_size),
                RandomOrientation(),
                #Noise(std=noise, items=[0]),
            ])
        )

        # Initiate standard DataLoader() instance
        dataloader = data.DataLoader(
            LIVECell_test_dset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )

        # Define loss function
        loss_func = nn.SmoothL1Loss()

        # Initiate FusionNet
        FusionNet = FusionGenerator(1,1,64).to(device=nn_handler_device, dtype=torch.float)

        # Load trained network
        model_path = f'{models_folder}FusionNet_snapshot{load_snapshot}.tar'
        checkpoint = torch.load(model_path)
        FusionNet.load_state_dict(checkpoint['model_module_state_dict'])
        print(f'\tTesting with snapshot of epoch {load_snapshot} of model {model} trained on {model_data_set}/{model_data_subset}...')
        print(f'\tUsing network to test on images from {data_set}/{data_subset}...')   
        print('\n\t', f'{print_separator}' * 71, '\n', sep='')
       
        # Wrap model for parallel GPU usage
        FusionNet = nn.DataParallel(
            FusionNet, 
            device_ids=gpu_device_ids
        )

        # Make sure evaluation mode is enabled
        FusionNet.eval()

        # Initialize loss log
        batch_loss_log = []
        
        # Perform online epoch image transformations
        LIVECell_trial.epoch_transform(LIVECell_test_dset)

        # Test FusionNet
        for iB, batch in enumerate(dataloader):
            
            # Wrap the batch, pass it forward and calculate loss
            x = Variable(batch['image']).to(device=nn_handler_device, dtype=torch.float)
            y_ = Variable(batch['annot']).to(device=nn_handler_device, dtype=torch.float)
            y = FusionNet(x)
            loss = loss_func(y, y_)

            # Record individual batch losses
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
        with open(f'{results_folder}FusionNet_test_loss_fromsnapshot{load_snapshot}.txt', 'w') as outfile:
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
                    f'{results_folder}FusionNet_image_snapshot{load_snapshot}_image{iI}.png'
                )
                v_utils.save_image(
                    y_[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_annot_snapshot{load_snapshot}_image{iI}.png'
                )
                v_utils.save_image(
                    y[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_pred_snapshot{load_snapshot}_image{iI}.png'
                )


# If test.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3 that take up memory
    os.system("""kill $(nvidia-smi | awk '$5=="PID" && $8>0 {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run test()
    test()
    print('\n\n', end='')