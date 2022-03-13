import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import torchvision.transforms   as transforms
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import Padding
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from utils                      import path_gen
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec


def test(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'trial',
    print_separator = '$',
    gpu_device_ids = getAvailable(
        limit=100,
        maxLoad=0.1,
        maxMemory=0.1
    ),
    model = '1',
    load_snapshot = 1,
    batch_size = 'max',
    pin_memory = True, 
    persistent_workers = True,
    save_images = 1
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
        print_separator (string): print output separation
            character (default = '$')
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier (default = 1)
        load_snapshot (int): training epoch age of saved 
            snapshot to test (default = 1)
        batch_size (int): data batch size (default = <maximum>)
        pin_memory (bool): tensors fetched by DataLoader pinned 
            in memory, enabling faster data transfer to 
            CUDA-enabled GPUs.(default = True)
        persistent_workers (bool): worker processes will not be 
            shut down after a dataset has been consumed once, 
            allowing the workers Dataset instances to remain 
            alive. (default = True)
        save_images (int): amount of test output images to save
            (default = 1; max = batch_size)
        
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

    # Assign devices if CUDA is available
    if torch.cuda.is_available(): 
        device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to test FusionNet')

    # Clip batch size if necessary
    if batch_size == 'max' or batch_size > 2 * len(gpu_device_ids):
        batch_size = 2 * len(gpu_device_ids)
        print(f'\tBatch size has been set to {batch_size}...')
    
    # Clip amount of images to be saved if necessary
    if save_images > batch_size:
        save_images = batch_size
        print(f'\tAmount of images to be saved has been set to {save_images}...')

    # Indicate how output will be saved
    print(f'\tAfter completion, {save_images} example outputs will be saved in path/results...')
    print('\n\t', f'{print_separator}' * 71, '\n', sep='')
    
    # Set number of workers equal to number of GPUs available
    num_workers = len(gpu_device_ids)

    # Generate folder path strings
    models_folder = path_gen([
        path,
        'models',
        data_set,
        data_subset,
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

    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_test_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            transform=transforms.Compose([
                RandomCrop(input_size=(520,704), output_size=512),
                RandomOrientation(),
                Padding(width=64),
                ToUnitInterval(),
                ToTensor()
            ])
        )

        # Initiate standard DataLoader() instance
        dataloader = data.DataLoader(
            LIVECell_test_dset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        # Initiate FusionNet
        FusionNet = nn.DataParallel(
            FusionGenerator(1,1,64), 
            device_ids=gpu_device_ids
        ).to(device=device, dtype=torch.float)

        # Load trained network
        FusionNet = torch.load(f'{models_folder}FusionNet_snapshot{load_snapshot}.pkl')
        print('\nTesting with snapshot of epoch {load_snapshot} of model {model}')
        print('\n\t', f'{print_separator}' * 71, '\n', sep='')

        # Define loss function
        loss_func = nn.SmoothL1Loss()
        
        # Test FusionNet
        batch_loss_log = []
        for iter, batch in enumerate(dataloader):
            
            # Wrap the batch and pass it forward
            x = Variable(batch['image']).to(device=device, dtype=torch.float)
            y_ = Variable(batch['annot']).to(device=device, dtype=torch.float)
            y = FusionNet(x)
        
            # Calculate the loss 
            loss = loss_func(y, y_)
                
            # Record individual batch losses
            batch_loss_log.append(loss.item())

            # Display progress
            batch_ratio = (iter) / (len(dataloader) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tBatches: [{:<{}}] {:.0f}%; Loss: {:.5f}".format(
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()

        # Display average loss
        print(f'\n\tAverage testing loss: {mean(batch_loss_log)}')

        # Loss log output
        with open(f'{results_folder}FusionNet_test_loss.txt', 'w') as outfile:
            for batch_loss in batch_loss_log:
                args = f'{batch_loss}\n'
                outfile.write(args)
        
        # Optional example network image outputs
        if save_images:
            for image in range(save_images):
                v_utils.save_image(
                    x[image].cpu().data, 
                    f'{results_folder}FusionNet_image_snapshot{load_snapshot}_image{image}.png'
                )
                v_utils.save_image(
                    y_[image].cpu().data, 
                    f'{results_folder}FusionNet_annot_snapshot{load_snapshot}_image{image}.png'
                )
                v_utils.save_image(
                    y[image].cpu().data, 
                    f'{results_folder}FusionNet_pred_snapshot{load_snapshot}_image{image}.png'
                )
    
    # Save testing parameters
    testing_variable_names = getargspec(test)[0]
    testing_parameters = getargspec(test)[3] 
    with open(f'{models_folder}FusionNet_testing_parameters{load_snapshot}.txt', 'w') as outfile:
        for iVn, iP in zip(testing_variable_names, testing_parameters):
            args = f'{iVn},{iP}\n'
            outfile.write(args)    


# If test.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3 that take up memory
    os.system("""kill $(nvidia-smi | awk '$5=="PID" && $8>0 {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run test()
    test()
    print('\n', end='')