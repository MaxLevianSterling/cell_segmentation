import os
import torch
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from FusionNet import * 
from datasets import LIVECell
from image_transforms import RandomCrop
from image_transforms import RandomOrientation
from image_transforms import BoundaryExtension
from image_transforms import Normalize
from image_transforms import ToTensor
from utils import path_gen


def test(
    path = 'C:/Users/Max-S/tndrg',
    data_set = 'LIVECell',
    data_subset = 'trial',
    model = '1',
    load_snapshot = 50,
    batch_size = 16,
    num_workers = 2, 
    pin_memory = True, 
    persistent_workers = True,
    loss_verbosity = True,
    save_images = 10
):  
    """Tests a FusionNet-type neural network quickly,
        without boosting or full deployment, just to 
        gauge network performance in terms of loss.
    
    Note:
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
        Neural network uses grayscale input

    Args:
        path (string): training data folder
        data_set (string): training data set 
        data_subset (string): training data subset 
        model (string): current model identifier
        load_snapshot (int): training epoch age of saved 
            snapshot to test (default = 50)
        batch_size (int): data batch size (default = 16)
        num_workers (int): enables multi-process data loading 
            with the specified number of loader worker 
            processes (default = 2) 
        pin_memory (bool): tensors fetched by DataLoader pinned 
            in memory, enabling faster data transfer to 
            CUDA-enabled GPUs.(default = True)
        persistent_workers (bool): worker processes will not be 
            shut down after a dataset has been consumed once, 
            allowing the workers Dataset instances to remain 
            alive. (default = True)
        loss_verbosity (bool): show average loss over testing
            set (default = True)
        save_images (int): amount of test output images to save
            (default = 10; max = batch_size)
        
    Returns:
        (Optional) Example network image outputs in the
            results subfolder
        A log of all testing set batch losses in the results 
            subfolder 
    """
    
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

    # Determine whether to use CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\tUsing {device.upper()} device')
    
    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_test_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            transform=transforms.Compose([
                RandomCrop(output_size=512),
                RandomOrientation(),
                BoundaryExtension(ext=64),
                Normalize(),
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

        # Initiate FusionNet | TODO: Implement DistributedDataParallel() instead?
        FusionNet = nn.DataParallel(FusionGenerator(1,1,64)).to(device) #.cuda() originally
        FusionNet = FusionNet.float()

        # Load trained network
        FusionNet = torch.load(f'{models_folder}FusionNet_snapshot{load_snapshot}.pkl')
        print('\nTesting with snapshot at epoch {load_snapshot} of model {model}')

        # Define loss function and optimizer
        loss_func = nn.SmoothL1Loss()
        
        # Test FusionNet
        loss_log = []
        for iter, batch in enumerate(dataloader):

            # Wrap the batch and pass it forward
            x = Variable(batch['image']).to(device)
            y_ = Variable(batch['annot']).to(device)
            y = FusionNet.forward(x.float())
            
            # Calculate the loss and note it
            loss_log.append(loss_func(y, y_))

        # Optional loss verbosity
        if loss_verbosity:
            print(f'Loss: {sum(loss_log)/len(loss_log)}')

        # Optional example network image outputs
        if save_images:
            for image in range(save_images):
                v_utils.save_image(
                    x[image].cpu().data, 
                    f'{results_folder}original_snapshot{load_snapshot}_image{image}.png'
                )
                v_utils.save_image(
                    y_[image].cpu().data, 
                    f'{results_folder}label_snapshot{load_snapshot}_image{image}.png'
                )
                v_utils.save_image(
                    y[image].cpu().data, 
                    f'{results_folder}gen_snapshot{load_snapshot}_image{image}.png'
                )
        
        # Loss log output
        with open(f'{results_folder}loss.txt', 'w') as outfile:
            for batch_loss in loss_log:
                args = f'{loss_log[batch_loss]}\n'
                outfile.write(args)

# Run test() if test.py is run directly
if __name__ == '__main__':
    test()