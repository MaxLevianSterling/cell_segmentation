import torch
import torch.nn.init    as init
from basic_blocks       import * 


class InceptionModule(nn.Module):

    def __init__(
        self, 
        act_fn,
        in_chan,
        out_chan,
    ):

        # Manage nn.Module inheritance
        super(InceptionModule, self).__init__()

        self.inc_one = inception_one(in_chan, out_chan)
        self.inc_three = inception_three(in_chan, out_chan)
        self.inc_five = inception_five(in_chan, out_chan)
        self.act_batch = act_batch(act_fn, out_chan)

    def forward(self, input):

        inc_one = self.inc_one(input)
        inc_three = self.inc_three(input)
        inc_five = self.inc_five(input)
        inc = torch.cat(
            (inc_one, inc_three, inc_five), 
            dim=1
        )
        inc_out = self.act_batch(inc)

        return inc_out


class Conv_residual_conv(nn.Module):
    """Controls one FusionNet residual layer"""

    def __init__(
        self, 
        direction, 
        act_fn, 
        small_chan, 
        large_chan,
        act_fn_output=None
    ):
        """ Args:
            direction (string): sampling direction
            act_fn (nn.Module): activation function
            in_chan (int): input channel depth
            out_chan (int): output channel depth
        """
    
        # Manage nn.Module inheritance
        super(Conv_residual_conv, self).__init__()

        # Define first convolutional block
        if direction == 'encoder' or direction == 'bridge':
            self.conv_in = InceptionModule(
                act_fn,
                small_chan, 
                large_chan, 
            )
        elif direction == 'decoder' or direction == 'out':
            self.conv_in = InceptionModule(
                act_fn,
                large_chan, 
                large_chan, 
            )
        elif direction == 'in':
            self.conv_in = conv_block(
                act_fn,
                small_chan, 
                large_chan, 
            )

        # Define residual block
        self.res_1 = InceptionModule(act_fn, large_chan, large_chan)
        self.res_2 = InceptionModule(act_fn, large_chan, large_chan)
        self.res_3 = InceptionModule(act_fn, large_chan, large_chan)

        # Define last convolutional block
        if direction == 'encoder' or direction == 'in':
            self.conv_out = InceptionModule(
                act_fn,
                large_chan, 
                large_chan, 
            )
        elif direction == 'decoder' or direction == 'bridge':
            self.conv_out = InceptionModule(
                act_fn,
                large_chan, 
                small_chan, 
            )
        elif direction == 'out':
            self.conv_out = conv_block(
                act_fn_output,
                large_chan, 
                small_chan, 
                no_batchnorm2d='yes',
            )

    def forward(self, input):
        """ Args:
            input (tensor): BxCxHxW input tensor
        """
        
        # Calculate residual layer output
        conv_in = self.conv_in(input)
        res_1 = self.res_1(conv_in)
        res_2 = self.res_2(res_1)
        res_3 = self.res_3(res_2)
        res = (conv_in + res_3) / 2
        conv_out = self.conv_out(res)
        
        return conv_out


class InfusionGenerator(nn.Module):
    """InfusionNet generator"""

    def __init__(
        self, 
        in_chan, 
        out_chan, 
        ngf,
        spat_drop_p,
        act_fn_encode,
        act_fn_decode,
        act_fn_output,
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
        super(InfusionGenerator, self).__init__()
        
        # Display status
        print('\tInitiating InfusionNet...')

        # Define encoder
        self.down_1 = Conv_residual_conv(
            'in',
            act_fn_encode,
            small_chan=in_chan, 
            large_chan=ngf, 
        )
        self.pool_1 = maxpool()
        self.drop_1 = spatial_dropout(spat_drop_p)
        self.down_2 = Conv_residual_conv(
            'encoder',
            act_fn_encode,
            small_chan=ngf, 
            large_chan=ngf * 2, 
        )
        self.pool_2 = maxpool()
        self.drop_2 = spatial_dropout(spat_drop_p)
        self.down_3 = Conv_residual_conv(
            'encoder',
            act_fn_encode,
            small_chan=ngf * 2, 
            large_chan=ngf * 4, 
        )
        self.pool_3 = maxpool()
        self.drop_3 = spatial_dropout(spat_drop_p)
        self.down_4 = Conv_residual_conv(
            'encoder',
            act_fn_encode,
            small_chan=ngf * 4, 
            large_chan=ngf * 8, 
        )
        self.pool_4 = maxpool()
        self.drop_4 = spatial_dropout(spat_drop_p)

        # Define bridge
        self.bridge = Conv_residual_conv(
            'bridge',
            act_fn_decode,
            small_chan=ngf * 8, 
            large_chan=ngf * 16, 
        )

        # Define decoder
        self.drop_5 = spatial_dropout(spat_drop_p)
        self.deconv_1 = conv_trans_block(ngf * 8)
        self.up_1 = Conv_residual_conv(
            'decoder',
            act_fn_decode,
            large_chan=ngf * 8, 
            small_chan=ngf * 4, 
        )
        self.drop_6 = spatial_dropout(spat_drop_p)
        self.deconv_2 = conv_trans_block(ngf * 4)
        self.up_2 = Conv_residual_conv(
            'decoder',
            act_fn_decode,
            large_chan=ngf * 4, 
            small_chan=ngf * 2, 
        )
        self.drop_7 = spatial_dropout(spat_drop_p)
        self.deconv_3 = conv_trans_block(ngf * 2)
        self.up_3 = Conv_residual_conv(
            'decoder',
            act_fn_decode,
            large_chan=ngf * 2, 
            small_chan=ngf, 
        )
        self.drop_8 = spatial_dropout(spat_drop_p)
        self.deconv_4 = conv_trans_block(ngf)
        self.up_4 = Conv_residual_conv(
            'out',
            act_fn_decode,
            large_chan=ngf, 
            small_chan=out_chan, 
            act_fn_output=act_fn_output,
        )

        # Initialize weights and biases
        with torch.no_grad():
            for iM, m in enumerate(self.modules()):
                if isinstance(m, nn.Conv2d):
                    # Choosing 'fan_out' preserves the magnitudes in the backwards pass.
                    # Backwards pass more chaotic because different celltypes, magnitudes
                    if iM >= 229:
                        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    else: 
                        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    init.zeros_(m.bias)
                elif isinstance(m, nn.ConvTranspose2d):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
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