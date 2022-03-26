import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    """Creates the basic FusionNet convolution block
    
    Args:
        in_dim (int): input channel depth
        out_dim (int): output channel depth
        act_fn (nn.Module): activation function
 
    Returns:
        (nn.Sequential()) Basic convolution block
    """

    block = nn.Sequential(
        nn.Conv2d(
            in_dim, 
            out_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            padding_mode='reflect'
        ),
        act_fn,
        nn.BatchNorm2d(out_dim)
    )
    return block


def conv_block_3(in_dim, out_dim, act_fn): 
    """Creates the combined FusionNet triple convolution block

    Args:
        in_dim (int): input channel depth
        out_dim (int): output channel depth
        act_fn (nn.Module): activation function
        
    Returns:
        (nn.Sequential()) Combined triple convolution block
    """

    block = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn)
    )
    return block
    
    
def maxpool():
    """Creates the basic FusionNet max pooling block

    Returns:
        (nn.Module) Basic max pooling block
    """

    block = nn.MaxPool2d(
        kernel_size=2, 
        stride=2, 
        padding=0
    )
    return block
    

def conv_trans_block(in_dim, out_dim, act_fn):
    """Creates the basic FusionNet upsampling block
    
    Args:
        in_dim (int): input channel depth
        out_dim (int): output channel depth
        act_fn (nn.Module): activation function

    Returns:
        (nn.Sequential()) Basic upsampling block
    """

    block = nn.Sequential(
        nn.ConvTranspose2d(
            in_dim, 
            out_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1
        ),
        act_fn,
        nn.BatchNorm2d(out_dim)
    )
    return block


def out_block(in_dim, out_dim, act_fn):
    """Creates the basic FusionNet output block
    
    Args:
        in_dim (int): input channel depth
        out_dim (int): output channel depth
        act_fn (nn.Module): activation function
 
    Returns:
        (nn.Sequential()) Basic output block
    """

    block = nn.Sequential(
        nn.Conv2d(
            in_dim, 
            out_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            padding_mode='reflect'
        ),
        act_fn,
    )
    return block