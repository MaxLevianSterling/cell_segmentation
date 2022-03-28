import os
import sys
import torch
import torch.utils.data         as data
import torchvision.utils        as v_utils
import torchvision.transforms   as transforms
import numpy                    as np
from torch.autograd             import Variable
from FusionNet                  import * 
from datasets                   import LIVECell
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import ToNormal
from image_transforms           import ToBinary
from image_transforms           import ToUnitInterval
from image_transforms           import LocalDeform
from image_transforms           import Noise
from image_transforms           import ToTensor
from utils                      import path_gen
from utils                      import get_gpu_memory
from utils                      import gpu_every_sec
from utils                      import optimizer_to
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec
from math                       import ceil


def train(
    path = '/mnt/sdg/maxs',
    data_set = 'LIVECell',
    data_subset = 'trial',
    print_separator = '$',
    gpu_device_ids = 'all_available',
    model = 'ReLuX_64_deform',
    load_snapshot = 250,
    pred_size = 64,
    orig_size = (520, 704),
    new_mean = .5,
    new_std = .15,
    localdeform = [6, 4],
    tobinary = .9,
    noise = .05,
    batch_size = 128,
    num_workers = 'ratio',
    pin_memory = True,
    persistent_workers = True,
    prefetch_factor = 2,
    drop_last = True,
    lr = .0002,
    weight_decay = .001,
    gamma = 0.99993068768,
    n_epochs = 250,
    save_snapshot_interval = 250,
    save_image_interval = 50,
    amp = False,
    reserved_gpus = []
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
            character
        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        model (string): current model identifier
        load_snapshot (int): training epoch age of saved
            snapshot to start training at (0 -> no snapshot 
            loading)
        localdeform (list of ints): [number of deforming 
            vectors along each axis (default = 12), maximum 
            vector magnitude (default = 8)
        tobinary (float): float to binary cutoff point 
            (default = .5)
        noise (float): standard deviation of Gaussian noise (
            default = .05)
        batch_size (int): data batch size (default = <maximum>)
        pin_memory (bool): tensors fetched by DataLoader pinned
            in memory, enabling faster data transfer to
            CUDA-enabled GPUs.
        persistent_workers (bool): worker processes will not be
            shut down after a dataset has been consumed once,
            allowing the workers Dataset instances to remain
            alive. 
        lr (float): network learning rate (default = .0002)
        n_epochs (int): number of training epochs 
        save_snapshot_interval (int): epoch interval with which
            network snapshot is saved 
        save_image_interval (int): epoch interval with which
            example output is saved

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

    # Save training parameters
    training_variable_names = getargspec(train)[0]
    training_parameters = getargspec(train)[3] 
    with open(f'{models_folder}FusionNet_training_parameters.txt', 'a') as outfile:
        for iVn, iP in zip(training_variable_names, training_parameters):
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
    print(f'\tFusionNet snapshots will be saved in path/models every {save_snapshot_interval} epochs...')
    print(f'\tEpoch losses will also be saved there every {save_snapshot_interval} epochs...')
    print(f'\tExample network outputs will be saved in path/results every {save_image_interval} epochs...')
    print('\n\t', f'{print_separator}' * 71, '\n', sep='')
   
    # Initiate custom DataSet() instance
    LIVECell_train_dset = LIVECell(
        path=path,
        data_set=data_set,
        data_subset=data_subset,
        dataset_device=dataset_device,
        offline_transforms=transforms.Compose([
            ToUnitInterval(),
            ToTensor(),
        ]),
        epoch_pretransforms=transforms.Compose([
            RandomCrop(input_size=orig_size, output_size=pred_size),
            RandomOrientation(),
            LocalDeform(size=localdeform[0], ampl=localdeform[1]),
            ToNormal(items=[0], new_mean=new_mean, new_std=new_std),
            ToBinary(cutoff=tobinary, items=[1]),
            Noise(std=noise, items=[0]),
        ])
    )

    # Reset number of workers if necessary
    if num_workers == 'max':
        num_workers = len(gpu_device_ids)
    elif num_workers == 'ratio':
        num_workers = min([
            ceil(len(LIVECell_train_dset)/batch_size), 
            len(gpu_device_ids)
        ])

    # Initiate standard DataLoader() instance
    dataloader = data.DataLoader(
        LIVECell_train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )

    # Define loss function 
    loss_func = nn.SmoothL1Loss()

    # # Initiate FusionNet
    # FusionNet = nn.DataParallel(
    #     FusionGenerator(1,1,64).to(device=nn_handler_device), 
    #     device_ids=gpu_device_ids,
    #     output_device=nn_handler_device
    # )

    # Initiate FusionNet
    FusionNet = FusionGenerator(1,1,64).to(device=nn_handler_device) 
        
    # for name, param in FusionNet.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #     break

    if load_snapshot:
        model_path = f'{models_folder}FusionNet_snapshot{load_snapshot}.tar'
        checkpoint = torch.load(model_path, map_location=nn_handler_device)
        check = FusionNet.load_state_dict(checkpoint['model_module_state_dict'])
    
    # for name, param in FusionNet.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #     break
    
    
    
    # for name, param in FusionNet.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #     break
    
    # Define optimizer and send to GPU
    optimizer = torch.optim.Adam(FusionNet.parameters(), lr=lr, weight_decay=weight_decay)

    # for param in optimizer.state.values():
    #     print(param)
    #     break

    if load_snapshot:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # for param in optimizer.state.values():
    #     print(param)
    #     break

    optimizer_to(optimizer, torch.device(nn_handler_device))
    
    # for param in optimizer.state.values():
    #     print(param)
    #     break

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    # Optional model snapshot loading
    if load_snapshot:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'\tSnapshot of model {model} at epoch {load_snapshot} restored...')
        print(f'\tUsing network to train on images from {data_set}/{data_subset}...')

    # Make sure training mode is enabled
    FusionNet.train()
    
    # Automatic mixed precision scaler
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    # Initialize loss log
    loss_log = []

    # Train FusionNet
    for iE in range(n_epochs):

        # Initialize batch loss log
        batch_loss_log = []

        # Perform online epoch image transformations
        LIVECell.epoch_pretransform(LIVECell_train_dset)

        # Use DataLoader() to get batch
        for iB, batch in enumerate(dataloader):

            # Set the gradients of the optimized tensor to zero
            optimizer.zero_grad()
            
            # Wrap the batch, pass it forward and calculate loss
            if amp:
                with torch.cuda.amp.autocast():
                    x = Variable(batch['image']).to(device=nn_handler_device)
                    y_ = Variable(batch['annot']).to(device=nn_handler_device)
                    y = FusionNet(x)
                    loss = loss_func(y, y_)
            else:
                x = Variable(batch['image']).to(device=nn_handler_device)
                y_ = Variable(batch['annot']).to(device=nn_handler_device)
                y = FusionNet(x)
                loss = loss_func(y, y_)

            # Pass the loss backwards
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update optimizer parameters
            if amp:
                scaler.step(optimizer)
            else:
                optimizer.step()

            # Updates scaler for next iteration
            if amp:
                scaler.update()
            
            # Record individual batch losses
            batch_loss_log.append(loss.item())

            # Display progress
            epoch_ratio = (iE) / (n_epochs - 1)
            batch_ratio = (iB + 1) / (len(dataloader))
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tEpochs: [{:<{}}] {:.0f}%; Batches: [{:<{}}] {:.0f}%; Loss: {:.5f}".format(
                    "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()
        
        # Update learning rate via scheduler
        scheduler.step()

        # Note the average loss of the epoch
        loss_log.append(mean(batch_loss_log))

        # At a regular interval
        if not (iE+1) % save_snapshot_interval:

            # Save FusionNet checkpoint
            torch.save({
                'model_module_state_dict': FusionNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'{models_folder}FusionNet_snapshot{load_snapshot+iE+1}.tar')

            # Save loss log
            with open(f'{results_folder}FusionNet_training_loss_fromsnapshot{load_snapshot}.txt', 'w') as outfile:
                for iE, epoch_loss in enumerate(loss_log):
                    args = f'{iE},{epoch_loss}\n'
                    outfile.write(args)
        
        # At a regular interval
        if not (iE+1) % save_image_interval:

            # Save example network image outputs
            iI = np.random.randint(0, list(batch.values())[0].shape[0])
            v_utils.save_image(
                x[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_image_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )
            v_utils.save_image(
                y_[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_annot_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )
            v_utils.save_image(
                y[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                f'{results_folder}FusionNet_pred_snapshot{load_snapshot+iE+1}_epoch{iE+1}.png'
            )


# If train.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    train()
    print('\n\n', end='')