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
from image_transforms           import ToUnitInterval
from image_transforms           import ToTensor
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import LocalDeform
from image_transforms           import ToNormal
from image_transforms           import ToBinary
from image_transforms           import Noise
from utils                      import path_gen
from utils                      import get_gpu_memory
from utils                      import gpu_every_sec
from utils                      import optimizer_to
from utils                      import reset_seeds
from math                       import ceil, floor
from statistics                 import mean
from GPUtil                     import getAvailable
from inspect                    import getargspec


def train(

    # Data variables
    path                = '/mnt/sdg/maxs',
    train_data_set      = 'LIVECell',
    train_data_subset   = 'trial',    
    val_data_set        = 'LIVECell',
    val_data_subset     = 'val_2',

    # GPU variables
    gpu_device_ids      = 'all_available',
    reserved_gpus       = [0, 1, 6, 7],

    # Training variables
    load_chkpt          = 0,
    n_epochs            = 100000,
    save_chkpt_interval = 1000,
    save_image_interval = 250,
    save_images         = 5,
    amp                 = True,

    # Validation variables
    validation_interval = 250,
    val_dataset_device  = 'cpu',

    # Model variables
    model               = 'fusion_128_base_new',
    in_chan             = 1,
    out_chan            = 1,
    ngf                 = 64,
    spat_drop_p         = .05,
    act_fn_encode       = nn.LeakyReLU(negative_slope=.1),
    act_fn_decode       = nn.LeakyReLU(negative_slope=.1),
    act_fn_output       = nn.LeakyReLU(negative_slope=.1),

    # Transform variables
    crop_size           = 128,
    orig_size           = (520, 704),
    localdeform         = [6, 4],
    new_mean            = 0.,
    new_std             = 1.,
    tobinary            = .5,
    noise               = .05,

    # DataLoader variables
    batch_size          = 32,
    shuffle             = True,
    drop_last           = True,

    # Optimizer and scheduler variables
    lr                  = .2,
    weight_decay        = 0.,
    #gamma               = 0.99993068768,
    gamma               = 1,
    sched_step_size     = 25,

    # Verbosity variables
    print_sep           = '$',
):
    """Trains a FusionNet-type neural network

    Note:
        Image data must be grayscale
        Required folder structure:
            <path>/
                data/
                    <train_data_set>/
                        images/
                            <train_data_subset>/<.tif files>
                        annotations/
                            <train_data_subset>/<.tif files>
                            (<.json file>)
                models/
                results/

    Args:
        path (string): path to training data folder
        train_data_set (string): training data set
        train_data_subset (string): training data subset

        gpu_device_ids (list of ints): gpu device ids used
            (default = <all available GPUs>)
        reserved_gpus (list of ints): reserved GPUs not to be
            used

        model (string): current model identifier
        load_chkpt (int): training epoch age of saved
            checkpoint to start training at (0 -> no checkpoint 
            loading)
        n_epochs (int): number of training epochs 
        save_chkpt_interval (int): epoch interval with which
            network checkpoint is saved 
        save_image_interval (int): epoch interval with which
            example output is saved
        amp (bool): whether automatic precision is to be used

        crop_size (int / tuple of ints): HxW output dimensions
            of crops 
        orig_size (int / tuple of ints): HxW input dimensions
            of original data 
        localdeform (list of ints): [number of deforming 
            vectors along each axis, maximum vector magnitude]
        new_mean (float): normalized mean of the input images
        new_std (float): normalized standard deviation of the 
            input images
        tobinary (float): float to binary cutoff point 
        noise (float): standard deviation of Gaussian noise 

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

        lr (float): initial network learning rate
        weight_decay (float): size of the L2 penalty for network
            weights
        gamma (float): ground number of epoch exponent with which
            scheduler updates network learning rate each scheduler
            step size
        sched_step_size (int): epoch step size with which to apply 
            gamma to learning rate

        print_sep (string): print output separation
            character

    Returns:
        FusionNet checkpoints in the models subfolder along with 
            a log of the loss over epochs and the training 
            parameters
        (Optional) Example network image outputs at various 
            training stages in the results subfolder

    Raises:
        RuntimeError: At least one GPU must be available to 
            train FusionNet
    """

    # Being beautiful is not a crime
    print('\n', f'{print_sep}' * 87, '\n', sep='')

    # Generate folder path strings
    models_folder = path_gen([
        path,
        'models',
        train_data_set,
        train_data_subset,
        model
    ])
    results_folder = path_gen([
        path,
        'results',
        train_data_set,
        train_data_subset,
        model,
        'training'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Check if model was already trained
    if os.path.isfile(f'{results_folder}FusionNet_training_loss_from_checkpoint{load_chkpt}.txt'):
        raise RuntimeError('Network duplicate detected')

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

    # Assign devices
    if torch.cuda.is_available(): 
        train_dataset_device = f'cuda:{gpu_device_ids[-1]}' 
        nn_handler_device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\tUsing GPU {gpu_device_ids[-1]} as online dataset storage...')
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to train FusionNet')
    
    # Indicate how output will be saved
    print(f'\tFusionNet checkpoints will be saved in path/models every {save_chkpt_interval} epochs...')
    print(f'\tEpoch losses will also be saved there every {save_chkpt_interval} epochs...')
    print(f'\tExample network outputs will be saved in path/results every {save_image_interval} epochs...')
    print('\n\t', f'{print_sep}' * 71, '\n', sep='')
   
    # Get training dataset
    LIVECell_train_dset = LIVECell(
        path=path,
        data_set=train_data_set,
        data_subset=train_data_subset,
        dataset_device=train_dataset_device,
        offline_transforms=Compose([
            ToUnitInterval(),
            ToTensor(),
        ]),
        epoch_pretransforms=Compose([
            RandomCrop(input_size=orig_size, output_size=crop_size),
            RandomOrientation(),
            LocalDeform(size=localdeform[0], ampl=localdeform[1]),
            ToNormal(items=[0], new_mean=new_mean, new_std=new_std, old_mean=.5021, old_std=.0449),
            ToBinary(cutoff=tobinary, items=[1]),
            Noise(std=noise, items=[0]),
        ])
    )

    # Get validation dataset
    LIVECell_val_dset = LIVECell(
        path=path,
        data_set=val_data_set,
        data_subset=val_data_subset,
        dataset_device=val_dataset_device,
        offline_transforms=Compose([
            ToUnitInterval(),
            ToTensor(),
            ToNormal(items=[0], new_mean=new_mean, new_std=new_std, old_mean=.5021, old_std=.0449),
        ]),
        epoch_pretransforms=Compose([
            RandomCrop(input_size=orig_size, output_size=crop_size),
            RandomOrientation(),
        ])
    )
      
    # Initiate custom data loader
    train_dataloader = GPU_dataloader(
        LIVECell_train_dset,
        dataset_device=train_dataset_device,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    # Initiate custom data loader
    val_dataloader = GPU_dataloader(
        LIVECell_val_dset,
        dataset_device=val_dataset_device,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    # Define loss function 
    loss_func = nn.SmoothL1Loss()

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
    
    # Load checkpoint and FusionNet
    if load_chkpt:
        chkpt_path = f'{models_folder}FusionNet_checkpoint{load_chkpt}.tar'
        chkpt = torch.load(chkpt_path, map_location=nn_handler_device)
        load_check = FusionNet.module.load_state_dict(chkpt['model_module_state_dict'])
        if not load_check:
            raise RuntimeError('\n\tNot all module parameters loaded correctly')
        print(f'\tCheckpoint of model {model} at epoch {load_chkpt} restored...')
        print(f'\tUsing network to train on images from {train_data_set}/{train_data_subset}...')

    # Define optimizer and send to GPU
    optimizer = torch.optim.SGD(FusionNet.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    #optimizer = torch.optim.Adam(FusionNet.parameters(), lr=lr, weight_decay=weight_decay)
    if load_chkpt:
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    optimizer_to(optimizer, torch.device(nn_handler_device))
    
    # Define scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.5, 
        patience=50, 
        threshold=0.01, 
        cooldown=10,
        min_lr=0.00005,
        verbose=True
    )
    if load_chkpt:
        scheduler.load_state_dict(chkpt['scheduler_state_dict'])

    # Make sure training mode is enabled
    FusionNet.train()
    
    # Automatic mixed precision scaler
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    # #Delete this
    # LIVECell.epoch_pretransform(LIVECell_train_dset)
    # batch = next(iter(train_dataloader))

    # Train FusionNet
    for iE in range(n_epochs):

        # Initialize training loss log
        train_loss_log = []

        # Perform online epoch image transformations
        LIVECell.epoch_pretransform(LIVECell_train_dset)

        # Use DataLoader() to get batch
        for iB, batch in enumerate(train_dataloader):

            # Set gradients to zero
            FusionNet.zero_grad()
            optimizer.zero_grad()
            
            # If automatic mixed precision is enabled
            if amp:
                with torch.cuda.amp.autocast():

                    # Wrap the batch and pass it forward
                    x = Variable(batch['image']).to(device=nn_handler_device)
                    y_ = Variable(batch['annot']).to(device=nn_handler_device, dtype=torch.float)
                    y = FusionNet(x)

                    # Calculate loss and pass it backwards
                    loss = loss_func(y, y_)
                    scaler.scale(loss).backward()

                    # Clip gradients
                    # torch.nn.utils.clip_grad_norm_(
                    #     FusionNet.parameters(), 
                    #     max_norm=1
                    # )

                    # Update optimizer
                    scaler.step(optimizer)

                    # Record scale and update
                    scale = scaler.get_scale()
                    scaler.update()

                    # Check for gradient overflow
                    optimizer_skipped = (scale > scaler.get_scale())
            else:

                # Wrap the batch and pass it forward
                x = Variable(batch['image']).to(device=nn_handler_device)
                y_ = Variable(batch['annot']).to(device=nn_handler_device, dtype=torch.float)
                y = FusionNet(x)
                
                # Calculate loss and pass it backwards
                loss = loss_func(y, y_)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    FusionNet.parameters(), 
                    max_norm=1
                )

                # Update optimizer
                optimizer.step()

            # Record individual batch losses
            train_loss_log.append(loss.item())

            # Display progress
            epoch_ratio = (iE) / (n_epochs - 1)
            batch_ratio = (iB + 1) / (len(train_dataloader))
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tEpochs: [{:<{}}] {:.0f}%; Batches: [{:<{}}] {:.0f}%; Loss: {:.5f}    ".format(
                    "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()

            # # Display progress
            # epoch_ratio = (iE) / (n_epochs - 1)
            # sys.stdout.write('\r')
            # sys.stdout.write(
            #     "\tEpochs: [{:<{}}] {:.0f}%; Loss: {:.5f}    ".format(
            #         "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
            #         loss.item()
            #     )
            # )
            # sys.stdout.flush()
        
        # Update learning rate via scheduler
        if not amp or amp and not optimizer_skipped:
            scheduler.step(mean(train_loss_log))

        # Save training loss
        with open(f'{results_folder}FusionNet_training_loss_from_checkpoint{load_chkpt}.txt', 'a') as outfile:
            args = f'{iE},{mean(train_loss_log)}\n'
            outfile.write(args)

        # At a regular interval
        if not (iE+1) % save_chkpt_interval:

            # Save FusionNet checkpoint
            torch.save({
                'model_module_state_dict': FusionNet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'{models_folder}FusionNet_checkpoint{load_chkpt+iE+1}.tar')
       
        # At a regular interval
        if not (iE+1) % save_image_interval:
            
            # Clip amount of images to be saved if necessary
            last_batch_size = list(batch.values())[0].shape[0]
            if save_images > last_batch_size:
                save_images = last_batch_size
                print(f'\tAmount of images to be saved has been reset to maximum {save_images}...')

            # Save example network image outputs
            image_idxs = torch.randperm(last_batch_size)
            for idx in range(save_images):
                iI = image_idxs[idx].item()
                v_utils.save_image(
                    x[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_image_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )
                v_utils.save_image(
                    y_[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_annot_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )
                v_utils.save_image(
                    y[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}FusionNet_pred_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )

        # At a regular interval
        if not (iE+1) % validation_interval:

            # Set model to evaluation mode
            FusionNet.eval()

            # Perform online epoch image transformations
            LIVECell.epoch_pretransform(LIVECell_val_dset)

            # Initialize validation loss log
            val_loss_log = []

            # Use DataLoader() to get batch
            for iB, batch in enumerate(val_dataloader):

                # Set gradients to zero
                FusionNet.zero_grad()
                optimizer.zero_grad()

                # If automatic mixed precision is enabled
                if amp:
                    with torch.cuda.amp.autocast():

                        # Wrap the batch and pass it forward
                        x = Variable(batch['image']).to(device=nn_handler_device)
                        y_ = Variable(batch['annot']).to(device=nn_handler_device, dtype=torch.float)
                        y = FusionNet(x)

                        # Calculate loss
                        loss = loss_func(y, y_)

                else:
                    # Wrap the batch and pass it forward
                        x = Variable(batch['image']).to(device=nn_handler_device)
                        y_ = Variable(batch['annot']).to(device=nn_handler_device, dtype=torch.float)
                        y = FusionNet(x)

                        # Calculate loss
                        loss = loss_func(y, y_)

                # Record individual batch losses
                val_loss_log.append(loss.item())

            # Save validation loss log
            with open(f'{results_folder}FusionNet_validation_loss_from_checkpoint{load_chkpt}.txt', 'a') as outfile:
                args = f'{iE},{mean(val_loss_log)}\n'
                outfile.write(args)

            # Set model to training mode
            FusionNet.train()


# If train.py is run directly
if __name__ == '__main__':
    
    # Kill all processes on GPU 2 and 3
    os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 2 && $2 <= 3 {print $5}')""")

    # Run train()
    train()
    print('\n\n', end='')