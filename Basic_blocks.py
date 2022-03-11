import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
        nn.BatchNorm2d(out_dim)
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn): #Why use learnable parameters here?It seems silly.
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn,
        nn.BatchNorm2d(out_dim)
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim, out_dim, act_fn): #Why no act_fn in 3rd layer originally?
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        #Dropout2d(p=0.1, inplace=True)
        conv_block(out_dim, out_dim, act_fn),
        #Dropout2d(p=0.1, inplace=True)
        conv_block(out_dim, out_dim, act_fn)
        # nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm2d(out_dim),
    )
    return model