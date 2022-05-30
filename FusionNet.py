import torch
import torch.nn.init    as init
from basic_blocks       import * 
from math               import sqrt


class Conv_residual_conv(nn.Module):
    """Controls one FusionNet residual layer"""

    def __init__(
        self, 
        direction, 
        act_fn, 
        small_chan, 
        large_chan,
        bn,
        act_after_bn,
        bn_momentum,
        act_fn_output=None,
    ):
        """ Args:
            direction (string): sampling direction
            act_fn (nn.Module): activation function
            small_chan (int): smallest channel depth 
                in residual block
            large_chan (int): largest channel depth 
                in residual block
            act_fn_output (nn.Module): activation function
                for the last residual block
        """
    
        # Manage nn.Module inheritance
        super(Conv_residual_conv, self).__init__()

        # Define first convolutional block
        if direction == 'encoder' or direction == 'bridge':
            self.conv_1 = conv_block(
                act_fn=act_fn,
                in_chan=small_chan, 
                out_chan=large_chan, 
                bn=bn, 
                act_after_bn=act_after_bn, 
                bn_momentum=bn_momentum,
            )
        elif direction == 'decoder' or direction == 'out':
            self.conv_1 = conv_block(
                act_fn=act_fn,
                in_chan=large_chan, 
                out_chan=large_chan, 
                bn=bn, 
                act_after_bn=act_after_bn, 
                bn_momentum=bn_momentum,
            )

        # Define residual block
        self.conv_2 = res_block(
            act_fn=act_fn,
            chan=large_chan, 
            bn=bn, 
            act_after_bn=act_after_bn, 
            bn_momentum=bn_momentum,
        )

        # Define last convolutional block
        if direction == 'encoder':
            self.conv_3 = conv_block(
                act_fn=act_fn,
                in_chan=large_chan, 
                out_chan=large_chan, 
                bn=bn, 
                act_after_bn=act_after_bn, 
                bn_momentum=bn_momentum,
            )
        elif direction == 'decoder' or direction == 'bridge':
            self.conv_3 = conv_block(
                act_fn=act_fn,
                in_chan=large_chan, 
                out_chan=small_chan, 
                bn=bn, 
                act_after_bn=act_after_bn, 
                bn_momentum=bn_momentum,
            )
        elif direction == 'out':
            self.conv_3 = conv_block(
                act_fn=act_fn_output,
                in_chan=large_chan, 
                out_chan=small_chan, 
                bn=False, 
                act_after_bn=False, 
                bn_momentum=bn_momentum,
            )

    def forward(self, input):
        """ Args:
            input (tensor): BxCxHxW input tensor
        """
        
        # Calculate residual layer output
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = (conv_1 + conv_2) / 2
        conv_3 = self.conv_3(res)
        
        return conv_3


class FusionGenerator(nn.Module):
    """FusionNet generator"""

    def __init__(
        self, 
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
    ):
        """ Args:
            in_chan (int): input channel depth
            out_chan (int): output channel depth
            ngf (int): feature depth factor
            spat_drop_p (float): spatial dropout chance
            act_fn_encode (nn.Module): activation function
                for encoder
            act_fn_decode (nn.Module): activation function
                for decoder            
            act_fn_output (nn.Module): activation function
                for output
        """
    
        # Manage nn.Module inheritance
        super(FusionGenerator, self).__init__()
        
        # Display status
        print('\tInitiating FusionNet...')

        # Define encoder
        self.down_1 = Conv_residual_conv(
            direction='encoder',
            act_fn=act_fn_encode,
            small_chan=in_chan, 
            large_chan=ngf, 
            bn=bn, 
            act_after_bn=act_after_bn, 
            bn_momentum=bn_momentum,
        )
        self.pool_1 = maxpool()
        self.drop_1 = spatial_dropout(spat_drop_p)
        self.down_2 = Conv_residual_conv(
            direction='encoder',
            act_fn=act_fn_encode,
            small_chan=ngf, 
            large_chan=ngf * 2,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.pool_2 = maxpool()
        self.drop_2 = spatial_dropout(spat_drop_p)
        self.down_3 = Conv_residual_conv(
            direction='encoder',
            act_fn=act_fn_encode,
            small_chan=ngf * 2, 
            large_chan=ngf * 4, 
            bn=bn, 
            act_after_bn=act_after_bn, 
            bn_momentum=bn_momentum,
        )
        self.pool_3 = maxpool()
        self.drop_3 = spatial_dropout(spat_drop_p)
        self.down_4 = Conv_residual_conv(
            direction='encoder',
            act_fn=act_fn_encode,
            small_chan=ngf * 4, 
            large_chan=ngf * 8,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.pool_4 = maxpool()
        self.drop_4 = spatial_dropout(spat_drop_p)

        # Define bridge
        self.bridge = Conv_residual_conv(
            direction='bridge',
            act_fn=act_fn_decode,
            small_chan=ngf * 8, 
            large_chan=ngf * 16,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )

        # Define decoder
        self.drop_5 = spatial_dropout(spat_drop_p)
        self.deconv_1 = conv_trans_block(
            act_fn=act_fn_decode, 
            chan=ngf * 8, 
            act_a_trans=act_a_trans, 
            bn_a_trans=bn_a_trans, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.up_1 = Conv_residual_conv(
            direction='decoder',
            act_fn=act_fn_decode,
            large_chan=ngf * 8, 
            small_chan=ngf * 4,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.drop_6 = spatial_dropout(spat_drop_p)
        self.deconv_2 = conv_trans_block(
            act_fn=act_fn_decode, 
            chan=ngf * 4, 
            act_a_trans=act_a_trans, 
            bn_a_trans=bn_a_trans, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.up_2 = Conv_residual_conv(
            direction='decoder',
            act_fn=act_fn_decode,
            large_chan=ngf * 4, 
            small_chan=ngf * 2,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.drop_7 = spatial_dropout(spat_drop_p)
        self.deconv_3 = conv_trans_block(
            act_fn=act_fn_decode, 
            chan=ngf * 2, 
            act_a_trans=act_a_trans, 
            bn_a_trans=bn_a_trans, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.up_3 = Conv_residual_conv(
            direction='decoder',
            act_fn=act_fn_decode,
            large_chan=ngf * 2, 
            small_chan=ngf,  
            bn=bn, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.drop_8 = spatial_dropout(spat_drop_p)
        self.deconv_4 = conv_trans_block(
            act_fn=act_fn_decode, 
            chan=ngf, 
            act_a_trans=act_a_trans, 
            bn_a_trans=bn_a_trans, 
            act_after_bn=act_after_bn,
            bn_momentum=bn_momentum,
        )
        self.up_4 = Conv_residual_conv(
            direction='out',
            act_fn=act_fn_decode,
            large_chan=ngf, 
            small_chan=out_chan,  
            bn=bn, 
            act_after_bn=act_after_bn,
            act_fn_output=act_fn_output,
            bn_momentum=bn_momentum,
        )

        #Initialize weights and biases
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    if init_name == 'XU':
                        init.xavier_uniform_(m.weight, gain=init.calculate_gain(init_gain, init_param))
                    if init_name == 'XN':
                        init.xavier_normal_(m.weight, gain=init.calculate_gain(init_gain, init_param))                    
                    if init_name == 'KU':
                        init.kaiming_uniform_(m.weight, mode=fan_mode, nonlinearity='relu')
                    if init_name == 'KN':
                        init.kaiming_normal_(m.weight, mode=fan_mode, nonlinearity='relu')                      
                    init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    init.normal_(m.weight, mean=1.0, std=0.02)
                    init.zeros_(m.bias)


    def forward(self, input):
        """ Args:
            input (tensor): BxCxHxW input tensor
        """

        # Encode
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        drop_1 = self.drop_1(pool_1)
        down_2 = self.down_2(drop_1)
        pool_2 = self.pool_2(down_2)
        drop_2 = self.drop_2(pool_2)
        down_3 = self.down_3(drop_2)
        pool_3 = self.pool_3(down_3)
        drop_3 = self.drop_3(pool_3)
        down_4 = self.down_4(drop_3)
        pool_4 = self.pool_4(down_4)
        drop_4 = self.drop_4(pool_4)

        # Bridge
        bridge = self.bridge(drop_4)

        # Decode
        drop_5 = self.drop_5(bridge)
        deconv_1 = self.deconv_1(drop_5)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)
        drop_6 = self.drop_6(up_1)
        deconv_2 = self.deconv_2(drop_6)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        drop_7 = self.drop_7(up_2)
        deconv_3 = self.deconv_3(drop_7)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        drop_8 = self.drop_8(up_3)
        deconv_4 = self.deconv_4(drop_8)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        return up_4