import torch
import torch.nn.init    as init
from Basic_blocks       import * 


class Conv_residual_conv(nn.Module):
    """Controls one FusionNet residual layer"""

    def __init__(self, in_dim, out_dim, act_fn):
        """Args:
            in_dim (int): input channel depth
            out_dim (int): output channel depth
            act_fn (nn.Module): activation function
        """
    
        # Manage nn.Module inheritance
        super(Conv_residual_conv, self).__init__()

        # Define class variables
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define residual layer
        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        """Args:
            input (tensor): BxCxHxW input tensor
        """
        
        # Calculate residual layer output
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = (conv_1 + conv_2) / 2
        conv_3 = self.conv_3(res)
        
        return conv_3


class FusionGenerator(nn.Module):
    """Control class generating FusionNet behaviour"""

    def __init__(
        self, 
        in_dim, 
        out_dim, 
        ngf
    ):
        """Args:
            in_dim (int): input channel depth
            out_dim (int): output channel depth
            ngf (int): feature depth factor
        """
    
        # Manage nn.Module inheritance
        super(FusionGenerator, self).__init__()
        
        # Initialize class variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ngf = ngf

        # Define activation function
        act_fn_encode = nn.LeakyReLU(0.2, inplace=True)        
        act_fn_decode = nn.ReLU()

        # Display status
        print('\tInitiating FusionNet...')

        # Define encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.ngf, act_fn_decode)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.ngf, self.ngf * 2, act_fn_decode)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.ngf * 2, self.ngf * 4, act_fn_decode)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.ngf * 4, self.ngf * 8, act_fn_decode)
        self.pool_4 = maxpool()

        # Define bridge
        self.bridge = Conv_residual_conv(self.ngf * 8, self.ngf * 16, act_fn_decode)

        # Define decoder
        self.deconv_1 = conv_trans_block(self.ngf * 16, self.ngf * 8, act_fn_decode)
        self.up_1 = Conv_residual_conv(self.ngf * 8, self.ngf * 8, act_fn_decode)
        self.deconv_2 = conv_trans_block(self.ngf * 8, self.ngf * 4, act_fn_decode)
        self.up_2 = Conv_residual_conv(self.ngf * 4, self.ngf * 4, act_fn_decode)
        self.deconv_3 = conv_trans_block(self.ngf * 4, self.ngf * 2, act_fn_decode)
        self.up_3 = Conv_residual_conv(self.ngf * 2, self.ngf * 2, act_fn_decode)
        self.deconv_4 = conv_trans_block(self.ngf * 2, self.ngf, act_fn_decode)
        self.up_4 = Conv_residual_conv(self.ngf, self.ngf, act_fn_decode)

        # Define output
        self.out = out_block(self.ngf, self.out_dim, nn.Tanh())

        # Initialize weights and biases
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # Choosing 'fan_out' preserves the magnitudes in the backwards pass.
                    # Backwards pass more chaotic because different celltypes, magnitudes
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                    # init.normal_(m.weight, mean=0.0, std=0.02)
                    init.zeros_(m.bias)
                elif isinstance(m, nn.ConvTranspose2d):
                    init.normal_(m.weight, mean=0.0, std=0.02)
                    init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    init.normal_(m.weight, mean=1.0, std=0.02)
                    init.zeros_(m.bias)


    def forward(self, input):
        """Args:
            input (tensor): BxCxHxW input tensor
        """

        # Encode
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        # Bridge
        bridge = self.bridge(pool_4)

        # Decode
        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        # Output
        out = self.out(up_4)

        return out