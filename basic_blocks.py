import torch.nn as nn


def inception_one(in_chan, out_chan):
    block = nn.Sequential(
        nn.Conv2d(
            in_chan, 
            int(out_chan/4), 
            kernel_size=1, 
            stride=1, 
            padding=0,
        ),
    )

    return block


def inception_three(in_chan, out_chan):
    block = nn.Sequential(
        # nn.Conv2d(
        #     in_chan, 
        #     int(in_chan/4), 
        #     kernel_size=1, 
        #     stride=1, 
        #     padding=0,
        # ),
        nn.Conv2d(
            int(in_chan), 
            int(out_chan/2), 
            kernel_size=3, 
            stride=1, 
            padding=1,
            padding_mode='reflect',
        ),
    )
    
    return block


def inception_five(in_chan, out_chan):
    block = nn.Sequential(
        # nn.Conv2d(
        #     in_chan, 
        #     int(in_chan/4), 
        #     kernel_size=1, 
        #     stride=1, 
        #     padding=0,
        # ),
        nn.Conv2d(
            int(in_chan), 
            int(out_chan/4), 
            kernel_size=5, 
            stride=1, 
            padding=2,
            padding_mode='reflect',
        ),
    )

    return block


def act_batch(act_fn, out_chan):
    """ Creates the basic FusionNet convolution 
        block
    
    Args:
        act_fn (nn.Module): activation function
        in_chan (int): input channel depth
        out_chan (int): output channel depth
 
    Returns:
        (nn.Sequential()) Basic convolution block
    """

    block = nn.Sequential(
        act_fn,
        nn.BatchNorm2d(out_chan),
    )

    return block
    
    
def conv_block(act_fn, in_chan, out_chan, no_batchnorm2d='', no_act_fn=''):
    """ Creates the basic FusionNet convolution 
        block
    
    Args:
        act_fn (nn.Module): activation function
        in_chan (int): input channel depth
        out_chan (int): output channel depth
 
    Returns:
        (nn.Sequential()) Basic convolution block
    """

    if no_batchnorm2d and no_act_fn:
        block = nn.Sequential(
            nn.Conv2d(
                in_chan, 
                out_chan, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                padding_mode='reflect'
            )
        )
    elif no_act_fn:
        block = nn.Sequential(
            nn.Conv2d(
                in_chan, 
                out_chan, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                padding_mode='reflect',
            ),
            nn.BatchNorm2d(out_chan),
        )
    elif no_batchnorm2d:
        block = nn.Sequential(
            nn.Conv2d(
                in_chan, 
                out_chan, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                padding_mode='reflect',
            ),
            act_fn,
        )
    else:
        block = nn.Sequential(
            nn.Conv2d(
                in_chan, 
                out_chan, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                padding_mode='reflect',
            ),
            act_fn,
            nn.BatchNorm2d(out_chan),
        )

    return block


def res_block(act_fn, chan): 
    """ Creates the combined FusionNet triple 
        convolution block

    Args:
        act_fn (nn.Module): activation function
        chan (int): channel depth
        
    Returns:
        (nn.Sequential()) Combined triple 
            convolution block
    """

    block = nn.Sequential(
        conv_block(act_fn, chan, chan),
        conv_block(act_fn, chan, chan),
        conv_block(act_fn, chan, chan),
    )

    return block
    
    
def maxpool():
    """ Creates the basic FusionNet max pooling 
        block

    Returns:
        (nn.Module) Basic max pooling block
    """

    block = nn.MaxPool2d(
        kernel_size=2, 
        stride=2, 
        padding=0,
    )

    return block


def spatial_dropout(spat_drop_p):
    """ Creates the basic FusionNet spatial dropout 
        block

    Args:
        spat_drop_p (float): spatial dropout chance

    Returns:
        (nn.Module) Basic spatial dropout block
    """

    block = nn.Dropout2d(p=spat_drop_p)

    return block
    

def conv_trans_block(chan):
    """ Creates the basic FusionNet upsampling block
    
    Args:
        chan (int): channel depth

    Returns:
        (nn.Sequential()) Basic upsampling block
    """

    block = nn.Sequential(
        nn.ConvTranspose2d(
            chan, 
            chan, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1
        ),
    )
    
    return block