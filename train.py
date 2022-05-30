import os
import sys
import torch
import torchvision.utils        as v_utils
from torch.autograd             import Variable
from FusionNet                  import *
from builders                   import Builder
from loaders                    import Loader
from image_transforms           import Compose
from image_transforms           import ToTensor
from image_transforms           import ToUnitInterval
from image_transforms           import RandomCrop
from image_transforms           import RandomOrientation
from image_transforms           import LocalDeform
from image_transforms           import ToNormal
from image_transforms           import ToBinary
from image_transforms           import Noise
from utils                      import path_gen
from utils                      import optimizer_to
from utils                      import get_gpu_list
from inspect                    import getargspec
from statistics                 import mean
from math                       import isnan, isinf


def train(

    # Data
    path                = '/mnt/sdg/maxs',
    annot_type          = 'soma', 
    
    # GPU
    n_gpus              = 4,
    kill_my_gpus        = False,
    reserved_gpus       = [4, 5, 6, 7],
    gpu_check_duration  = 5,
    gpu_usage_limit     = 500,

    # Training
    train_data_set      = 'LIVECell',
    train_data_type     = 'part_set',
    train_data_subset   = '5',    
    train_subset_type   = 'train',
    save_model_name     = 'model_name',
    n_epochs            = 1000,
    supervised          = True,
    clip_gradients      = False,
    max_grad_norm       = 1,

    # Validation
    val_data_set        = 'LIVECell',
    val_data_type       = 'part_set',
    val_data_subset     = '5',
    val_subset_type     = 'val',
    val_dataset_device  = 'cpu',
    validation_interval = 10,

    # Loading
    load_data_set       = 'LIVECell',
    load_data_type      = '',
    load_data_subset    = '',    
    load_subset_type    = '',
    load_model_name     = 'pretrain',
    load_chkpt          = 0,

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
    localdeform         = [6, 8],
    new_mean            = 0.,
    new_std             = 1.,
    tobinary            = .5,
    noise               = .25,

    # Loader
    batch_size          = 32,
    shuffle             = True,
    drop_last           = True,

    # Loss, optimizer, and scheduler
    loss_name           = 'SL1',
    optim_name          = 'Adam',
    weight_decay        = .00001,
    initial_lr          = .0128,
    schd_min_lr         = .0002,
    schd_decrease_f     = .5, 
    schd_patience       = 100, 
    schd_threshold      = .01, 
    schd_cooldown       = 10,

    # Output
    print_sep           = '$',
    schd_verbose        = True,
    save_chkpt_interval = 500,
    save_image_interval = 100,
    n_images_saved      = 3,
):
    """Train a cell segmentation network

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

        Training
        train_data_set (string): training data set (i.e.
            LIVECell)
        train_data_type (string): training data type (
            i.e. per_celltype, part_set, full_set)
        train_data_subset (string): training data subset (
            e.g. BV2, 50%)
        train_subset_type (string): training data subset type
            (i.e. train, test, val)
        save_model_name (string): new model identifier
        n_epochs (int): number of training epochs 
        supervised (bool): whether training should be 
            supervised with the specified annotation style
            or self-supervised as an autoencoder
        clip_gradients (bool): whether to clip gradients
        max_grad_norm (int): what norm to clip gradients at

        Validation
        val_data_set (string): validation data set ( i.e.
            LIVECell)
        val_data_type (string): validation data type (
            i.e. per_celltype, part_set, full_set)
        val_data_subset (string): validation data subset (
            e.g. BV2, 50%)
        val_subset_type (string): validation data subset 
            type (i.e. train, test, val)
        val_dataset_device (string): what device to handle 
            validation data on
        validation_interval (int): epoch interval with which 
            validation is performed

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
        localdeform (list of ints): [number of deforming 
            vectors along each axis, maximum vector magnitude]
        new_mean (float): normalized mean of the input images
        new_std (float): normalized standard deviation of the 
            input images
        tobinary (float): float to binary cutoff point 
        noise (float): standard deviation of Gaussian noise 

        Loader
        batch_size (int): data batch size
        shuffle (bool): whether input data is to be shuffled
        drop_last (bool): whether to use the last unequally 
            sized batch per dataset consumption
        
        Optimizer and scheduler
        weight_decay (float): size of the L2 penalty for network
            weights
        initial_lr (float): initial network learning rate
        schd_min_lr (float): minimal network learning rate
        schd_decrease_f (float): plateau learning rate decrease
            factor
        schd_patience (int): how many epochs to observe the
            minimal loss decrease for before lowering the
            learning rate
        schd_threshold (int): the minimal loss decrease factor 
            not yet designated as a plateau
        schd_cooldown (int): amount of epochs to wait after 
            learning rate decrease 
        
        Output
        print_sep (string): print output separation
            character
        schd_verbose (bool): whether to output scheduler updates
        save_chkpt_interval (int): epoch interval with which
            network checkpoint is saved 
        save_image_interval (int): epoch interval with which
            example output is saved
        n_images_saved (int): number of images saved every 
            save_image_interval

    Returns:
        Model checkpoints in save_model_folder 
        Training parameters in save_model_folder 
        Training loss log in results_folder 
        Validation loss log in results_folder 
        (Optional) Image outputs in results_folder

    Raises:
        RuntimeError: At least one GPU must be available to 
            train model
    """

    # Being beautiful is not a crime
    print('\n', f'{print_sep}' * 87, '\n', sep='')

    # Generate folder path strings
    save_model_folder = path_gen([
        path,
        'models',
        train_data_set,
        train_data_type,
        train_data_subset,
        train_subset_type,
        annot_type,
        save_model_name
    ])
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
        train_data_set,
        train_data_type,
        train_data_subset,
        train_subset_type,
        annot_type,
        save_model_name,
        'training'
    ])
 
    # Create output directories
    if not os.path.isdir(save_model_folder):
        os.makedirs(save_model_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    # Avoid network overwrite
    if os.path.isfile(f'{results_folder}training_loss_from_checkpoint{load_chkpt}.txt'):
        raise RuntimeError('Network duplicate detected')

    # Save training parameters
    training_parameter_names = getargspec(train)[0]
    training_parameters = getargspec(train)[3] 
    with open(f'{save_model_folder}training_parameters.txt', 'w') as outfile:
        for iVn, iP in zip(training_parameter_names, training_parameters):
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
        train_dataset_device = f'cuda:{gpu_device_ids[-1]}' 
        nn_handler_device = f'cuda:{gpu_device_ids[0]}' 
        print(f'\n\tUsing GPU {gpu_device_ids[-1]} as online dataset storage...')
        print(f'\tUsing GPU {gpu_device_ids[0]} as handler for GPUs {gpu_device_ids}...')
    else: 
        raise RuntimeError('\n\tAt least one GPU must be available to train model')
    
    # Indicate how output will be saved
    print(f'\tModel checkpoints will be saved in path/models every {save_chkpt_interval} epochs...')
    print(f'\tExample network outputs will be saved in path/results every {save_image_interval} epochs...')
    print(f'\tValidation losses will be saved there every {validation_interval} epochs...')
    print(f'\tTraining losses will be saved there continuously, as well...')
    print('\n\t', f'{print_sep}' * 71, '\n', sep='')
   
    # Get training dataset and loader
    train_dset = Builder(
        supervised=supervised,
        path=path,
        data_set=train_data_set,
        data_type=train_data_type,
        data_subset=train_data_subset,
        subset_type=train_subset_type,
        annot_type=annot_type,
        dataset_device=train_dataset_device,
        supervised_offline_transforms=Compose([
            ToTensor(type=torch.float),
            ToUnitInterval(items=[0, 1]),
            ToNormal(items=[0], new_mean=new_mean, new_std=new_std),
        ]),
        epoch_pretransforms=Compose([
            RandomCrop(input_size=orig_size, output_size=crop_size),
            RandomOrientation(),
            LocalDeform(size=localdeform[0], ampl=localdeform[1]),
            ToBinary(cutoff=tobinary, items=[1]),
            Noise(std=noise, items=[0]),
        ]),
        unsupervised_offline_transforms=Compose([
            ToTensor(type=torch.float),
            ToUnitInterval(items=[0, 1]),
            ToNormal(items=[0, 1], new_mean=new_mean, new_std=new_std),
        ]),
    )
    train_loader = Loader(
        train_dset,
        dataset_device=train_dataset_device,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    # Get validation dataset and loader
    if supervised:
        val_dset = Builder(
            path=path,
            supervised=supervised,
            data_set=val_data_set,
            data_type=val_data_type,
            data_subset=val_data_subset,
            subset_type=val_subset_type,
            annot_type=annot_type,
            dataset_device=val_dataset_device,
            supervised_offline_transforms=Compose([
                ToTensor(type=torch.float),
                ToUnitInterval(items=[0, 1]),
                ToNormal(items=[0], new_mean=new_mean, new_std=new_std),
            ]),
            epoch_pretransforms=Compose([
                RandomCrop(input_size=orig_size, output_size=crop_size),
                RandomOrientation(),
            ])
        )
        val_loader = Loader(
            val_dset,
            dataset_device=val_dataset_device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    # Define loss function 
    if loss_name == 'SL1':
        loss_func = nn.SmoothL1Loss()
    elif loss_name == 'L1':
        loss_func = nn.L1Loss()
    elif loss_name == 'MSE':
        loss_func = nn.MSELoss()

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
    
    # Load checkpoint and model
    if load_chkpt:
        chkpt_path = f'{load_model_folder}checkpoint{load_chkpt}.tar'
        chkpt = torch.load(chkpt_path, map_location=nn_handler_device)
        load_check = Model.module.load_state_dict(chkpt['model_module_state_dict'])
        if not load_check:
            raise RuntimeError('\n\tNot all module parameters loaded correctly')
        print(f'\tCheckpoint of model {load_model_name} at epoch {load_chkpt} restored...')

    # Define optimizer and send to GPU
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(Model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(Model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if load_chkpt:
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    optimizer_to(optimizer, torch.device(nn_handler_device))
    
    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=schd_decrease_f, 
        patience=schd_patience, 
        threshold=schd_threshold, 
        cooldown=schd_cooldown,
        min_lr=schd_min_lr,
        verbose=schd_verbose,
    )
    if load_chkpt:
        scheduler.load_state_dict(chkpt['scheduler_state_dict'])

    # Train
    Model.train()
    scaler = torch.cuda.amp.GradScaler()
    for iE in range(n_epochs):
        train_loss_log = []
        Builder.epoch_pretransform(train_dset)
        for iB, batch in enumerate(train_loader):

            # Set gradients to zero
            Model.zero_grad()
            optimizer.zero_grad()
            
            # With automatic mixed precision
            with torch.cuda.amp.autocast():

                # Wrap the batch and pass it forward
                x = Variable(batch['image']).to(device=nn_handler_device)
                y_ = Variable(batch['comp']).to(device=nn_handler_device, dtype=torch.float)
                y = Model(x)

                # Calculate loss and pass it backwards
                loss = loss_func(y, y_)
                scaler.scale(loss).backward()

                # Stop if gradients have exploded
                if loss.item() > 10000 or isnan(loss.item()) or isinf(loss.item()):
                    return

                # Clip gradients 
                if clip_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        Model.parameters(), 
                        max_norm=max_grad_norm,
                    )

                # Update optimizer
                scaler.step(optimizer)

                # Record scale and update
                scale = scaler.get_scale()
                scaler.update()

                # Check for gradient overflow
                optimizer_skipped = (scale > scaler.get_scale())

            # Record individual batch losses
            train_loss_log.append(loss.item())

            # Display progress
            epoch_ratio = (iE) / (n_epochs - 1)
            batch_ratio = (iB + 1) / (len(train_loader))
            sys.stdout.write('\r')
            sys.stdout.write(
                "\tEpochs: [{:<{}}] {:.0f}%; Batches: [{:<{}}] {:.0f}%; Loss: {:.5f}    ".format(
                    "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
                    "=" * int(20*batch_ratio), 20, 100*batch_ratio,
                    loss.item()
                )
            )
            sys.stdout.flush()
       
        # Update learning rate via scheduler
        if not optimizer_skipped:
            scheduler.step(mean(train_loss_log))

        # Save training loss
        with open(f'{results_folder}training_loss_from_checkpoint{load_chkpt}.txt', 'a') as outfile:
            args = f'{iE},{mean(train_loss_log)}\n'
            outfile.write(args)

        # At a regular interval
        if not (iE+1) % save_chkpt_interval:

            # Save checkpoint
            torch.save({
                'model_module_state_dict': Model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'{save_model_folder}checkpoint{load_chkpt+iE+1}.tar')
       
        # At a regular interval
        if not (iE+1) % save_image_interval:
            
            # Clip amount of images to be saved if necessary
            last_batch_size = list(batch.values())[0].shape[0]
            if n_images_saved > last_batch_size:
                n_images_saved = last_batch_size
                print(f'\tAmount of images to be saved has been reset to maximum {n_images_saved}...')

            # Save example network image outputs
            image_idxs = torch.randperm(last_batch_size)
            for idx in range(n_images_saved):
                iI = image_idxs[idx].item()
                v_utils.save_image(
                    x[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}image_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )
                v_utils.save_image(
                    y_[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}comp_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )
                v_utils.save_image(
                    y[iI, 0:1, :, :].detach().to('cpu').type(torch.float32),
                    f'{results_folder}pred_checkpoint{load_chkpt}_epoch{iE+1}_image{iI}.png'
                )

        # Validate 
        if supervised and not (iE+1) % validation_interval:
            with torch.no_grad():
                Model.eval()
                Builder.epoch_pretransform(val_dset)
                val_loss_log = []
                for iB, batch in enumerate(val_loader):

                    # Set gradients to zero
                    Model.zero_grad()
                    optimizer.zero_grad()

                    # With automatic mixed precision
                    with torch.cuda.amp.autocast():

                        # Wrap the batch and pass it forward
                        x = Variable(batch['image']).to(device=nn_handler_device)
                        y_ = Variable(batch['comp']).to(device=nn_handler_device, dtype=torch.float)
                        y = Model(x)

                        # Calculate loss
                        loss = loss_func(y, y_)

                    # Record individual batch losses
                    val_loss_log.append(loss.item())

                # Save validation loss log
                with open(f'{results_folder}validation_loss_from_checkpoint{load_chkpt}.txt', 'a') as outfile:
                    args = f'{iE},{mean(val_loss_log)}\n'
                    outfile.write(args)

                # Set model back to training mode
                Model.train()


# If train.py is run directly
if __name__ == '__main__':
    
    # Run train()
    train()
    print('\n\n', end='')