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
from image_transforms           import LocalDeform
from image_transforms           import Padding
from image_transforms           import ToUnitInterval
from image_transforms           import ToBinary
from image_transforms           import Noise
from image_transforms           import ToTensor
from utils                      import path_gen
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec


def train(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'train',
    print_separator = '$',
    gpu_device_ids = getAvailable(
        limit=100, 
        maxLoad=0.1, 
        maxMemory=0.1
    ),
    model = '2',
    load_snapshot = 150,
    batch_size = 'max',
    pin_memory = True,
    persistent_workers = True,
    lr = .0002,
    n_epochs = 1000,
    save_snapshot_interval = 25,
    save_image_interval = 25
):
    """Trains a FusionNet-type neural network

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
        path (string): path to training data folder
        data_set (string): training data set
        data_subset (string): training data subset
        print_separator (string): print output separation
            character (default = '$')
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier (default = 1)
        load_snapshot (int): training epoch age of saved
            snapshot to start training at (default = 0
            -> no snapshot loading)
        batch_size (int): data batch size (default = <maximum>)
        pin_memory (bool): tensors fetched by DataLoader pinned
            in memory, enabling faster data transfer to
            CUDA-enabled GPUs.(default = True)
        persistent_workers (bool): worker processes will not be
            shut down after a dataset has been consumed once,
            allowing the workers Dataset instances to remain
            alive. (default = True)
        lr (float): network learning rate (default = .0002)
        n_epochs (int): number of training epochs 
            (default = 10,000)
        save_snapshot_interval (int): epoch interval with which
            network snapshot is saved (default = 1,000)
        save_image_interval (int): epoch interval with which
            example output is saved (default = 1,000)

    Returns:
        FusionNet snapshots in the models subfolder along with 
            a log of the loss over epochs and the training 
            parameters
        (Optional) Example network image outputs at various 
            training stages in the results subfolder

    Raises:
        RuntimeError: At least one GPU must be available to 
            train FusionNet
    """

    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 87, '\n', sep='')

    # Assign devices if CUDA is available
    if torch.cuda.is_available(): 
        device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to train FusionNet')

    # Clip batch size if necessary
    if batch_size == 'max' or batch_size > 2 * len(gpu_device_ids):
        batch_size = 2 * len(gpu_device_ids)
        print(f'\tBatch size has been set to {batch_size}...')
        print('\n\t', f'{print_separator}' * 71, '\n', sep='')
    
    # Indicate how output will be saved
    print(f'\tFusionNet snapshots will be saved in path/models every {save_snapshot_interval} epochs...')
    print(f'\tEpoch losses will also be saved there every {save_snapshot_interval} epochs...')
    print(f'\tExample network outputs will be saved in path/results every {save_image_interval} epochs...')
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
        'training'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Initiate custom DataSet() instance
    LIVECell_train_dset = LIVECell(
        path=path,
        data_set=data_set,
        data_subset=data_subset,
        transform=transforms.Compose([
            RandomCrop(input_size=(520,704), output_size=512),
            RandomOrientation(),
            LocalDeform(size=12, ampl=8),
            Padding(width=64),
            ToUnitInterval(),
            ToBinary(cutoff=.5, items=[1]),
            Noise(std=.05, items=[0]),
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

    # Initiate FusionNet
    FusionNet = nn.DataParallel(
        FusionGenerator(1,1,64), 
        device_ids=gpu_device_ids
    ).to(device=device, dtype=torch.float)

    # Optional model snapshot loading
    if load_snapshot:
        model_path = f'{models_folder}FusionNet_snapshot{load_snapshot}.pkl'
        FusionNet.load_state_dict(torch.load(model_path))
        print(f'\tSnapshot of model {model} at epoch {load_snapshot} restored...')
        print(f'\tUsing network to train on images from {data_set}/{data_subset}...')

    # Define loss function and optimizer
    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(FusionNet.parameters(), lr=lr)
    
    # Automatic mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Train FusionNet
    loss_log = []
    for iE in range(n_epochs):
        batch_loss_log = []
        for iter, batch in enumerate(dataloader):

            # Set the gradients of the optimized tensor to zero
            optimizer.zero_grad()
            
            # Automatic mixed precision casting
            with torch.cuda.amp.autocast():

                # Wrap the batch and pass it forward
                x = Variable(batch['image']).to(device=device, dtype=torch.float)
                y_ = Variable(batch['annot']).to(device=device, dtype=torch.float)
                y = FusionNet(x)

                # Calculate the loss 
                loss = loss_func(y, y_)

            # Pass the loss backwards
            scaler.scale(loss).backward()

            # Update optimizer parameters
            scaler.step(optimizer)
            
            # Updates scaler for next iteration
            scaler.update()

            # Record individual batch losses
            batch_loss_log.append(loss.item())
            
            # Display progress
            epoch_ratio = (iE) / (n_epochs - 1)
            batch_ratio = (iter) / (len(dataloader) - 1)
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tEpochs: [{:<{}}] {:.0f}%; Batches: [{:<{}}] {:.0f}%; Loss: {:.5f}".format(
                    "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()

        # Note the average loss of the epoch
        loss_log.append(mean(batch_loss_log))

        # FusionNet snapshot saving along with loss log
        if not (iE+1) % save_snapshot_interval:
            torch.save(
                FusionNet.state_dict(), 
                f'{models_folder}FusionNet_snapshot{load_snapshot+iE+1}.pkl'
            )
            with open(f'{results_folder}FusionNet_training_loss_epoch{iE+1}.txt', 'w') as outfile:
                for epoch_loss in loss_log:
                    args = f'{epoch_loss}\n'
                    outfile.write(args)
        
        # Optional example network image outputs
        if not (iE+1) % save_image_interval:
            v_utils.save_image(
                x[0].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_image_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )
            v_utils.save_image(
                y_[0].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_annot_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )
            v_utils.save_image(
                y[0].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_pred_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )

    # Save training parameters
    training_variable_names = getargspec(train)[0]
    training_parameters = getargspec(train)[3] 
    with open(f'{models_folder}FusionNet_training_parameters{load_snapshot+iE+1}.txt', 'w') as outfile:
        for iVn, iP in zip(training_variable_names, training_parameters):
            args = f'{iVn},{iP}\n'
            outfile.write(args)


# If train.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    train()
    print('\n', end='')