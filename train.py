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
from image_transforms import LocalDeform
from image_transforms import BoundaryExtension
from image_transforms import Normalize
from image_transforms import Noise
from image_transforms import ToTensor
from utils import path_gen


def train(
    path = 'C:/Users/Max-S/tndrg',
    data_set = 'LIVECell',
    data_subset = 'trial',
    model = '1',
    load_snapshot = 0,
    save_snapshots = True,
    batch_size = 16,
    num_workers = 2, 
    pin_memory = True, 
    persistent_workers = True,
    lr = .0002,
    epochs = 50,
    verbosity_interval = 1,
    save_image_interval = 10,
    save_snapshot_interval = 1000,
):  
    """Trains a FusionNet-type neural network
    
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
            snapshot to start training at (default = 0 
            -> no snapshot loading)
        save_snapshots (bool): whether to save network
            snapshots at the specified interval (default = True)
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
        lr (float): network learning rate (default = .0002)
        epochs (int): number of training epochs (default = 50)
        verbosity_interval (int): epoch interval with which loss
            is displayed (default = 1)
        save_image_interval (int): epoch interval with which 
            example output is saved (default = 10)
        save_snapshot_interval (int): epoch interval with which 
            network snapshot is saved (default = 1000)
        
    Returns:
        (Optional) Example network image outputs at various training 
            stages in the results subfolder
        (Optional) FusionNet snapshots in the models subfolder along 
            with a log of the loss over epochs
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
        'training'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Determine whether to use CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\tUsing {device.upper()} device')

    # Initiate custom DataSet() instance
    LIVECell_train_dset = LIVECell(
        path=path,
        data_set=data_set,
        data_subset=data_subset,
        transform=transforms.Compose([
            RandomCrop(output_size=512),
            RandomOrientation(),
            LocalDeform(size=12,ampl=8), 
            BoundaryExtension(ext=64),
            Normalize(),
            Noise(std=.1),
            ToTensor()
        ])
    )

    # Initiate standard DataLoader() instance
    dataloader = data.DataLoader(
        LIVECell_train_dset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # Initiate FusionNet | TODO: Implement DistributedDataParallel() instead?
    FusionNet = nn.DataParallel(FusionGenerator(1,1,64)).to(device) #.cuda() originally
    FusionNet = FusionNet.float()

    # Optional model snapshot loading
    if load_snapshot:
        FusionNet = torch.load(f'{models_folder}FusionNet_snapshot{load_snapshot}.pkl')
        print('\nSnapshot of model {model} at epoch {load_snapshot} restored')

    # Define loss function and optimizer
    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(FusionNet.parameters(), lr=lr)
    
    # Train FusionNet
    loss_log = []
    for epoch in range(epochs):
        for iter, batch in enumerate(dataloader):
            
            # Set the gradients of the optimized tensor to zero
            optimizer.zero_grad()

            # Wrap the batch and pass it forward
            x = Variable(batch['image']).to(device)
            y_ = Variable(batch['annot']).to(device)
            y = FusionNet.forward(x.float())
            
            # Calculate the loss and pass it backward
            loss = loss_func(y, y_)
            loss.backward()
            optimizer.step()

        # Note the loss of the epoch
        loss_log.append(loss)

        # Optional loss verbosity
        if epoch % verbosity_interval == 0:
            print(f'Epoch: {epoch+1}/{epochs}; Loss: {loss}')

        # Optional example network image outputs
        if epoch % save_image_interval == 0:
            v_utils.save_image(
                x[0].cpu().data, 
                f'{results_folder}original_snapshot{load_snapshot+epoch}_epoch{epoch}.png'
            )
            v_utils.save_image(
                y_[0].cpu().data, 
                f'{results_folder}label_snapshot{load_snapshot+epoch}_epoch{epoch}.png'
            )
            v_utils.save_image(
                y[0].cpu().data, 
                f'{results_folder}gen_snapshot{load_snapshot+epoch}_epoch{epoch}.png'
            )
        
        # Optional FusionNet snapshot saving along with loss log
        if save_snapshots and epoch % save_snapshot_interval == 0:
            torch.save(FusionNet, f'{models_folder}FusionNet_snapshot{load_snapshot+epoch}.pkl')    
            with open(f'{results_folder}loss_epoch{epoch}.txt', 'w') as outfile:
                for epoch_loss in loss_log:
                    args = f'{loss_log[epoch_loss]}\n'
                    outfile.write(args)

# Run train() if train.py is run directly
if __name__ == '__main__':
    train()