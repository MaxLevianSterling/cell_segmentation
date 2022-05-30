from train import train
import torch.nn as nn


# Parameters
n_gpus = 4

# Networks
network_names = ['D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W']

ngf = [6.40E+01,3.20E+01,3.20E+01,1.28E+02,9.60E+01,6.40E+01,1.28E+02,1.28E+02,9.60E+01,6.40E+01,9.60E+01,6.40E+01,9.60E+01,1.28E+02,6.40E+01,3.20E+01,6.40E+01,1.28E+02,6.40E+01,6.40E+01]
bs_per_gpu = [8.00E+00,4.00E+00,1.60E+01,8.00E+00,2.00E+00,1.60E+01,2.00E+00,2.00E+00,4.00E+00,1.60E+01,8.00E+00,2.00E+00,2.00E+00,8.00E+00,8.00E+00,8.00E+00,2.00E+00,4.00E+00,8.00E+00,2.00E+00]
crop_size = [1.92E+02,1.28E+02,3.20E+02,1.28E+02,1.28E+02,1.92E+02,2.56E+02,3.20E+02,3.20E+02,1.92E+02,1.92E+02,2.56E+02,1.92E+02,1.28E+02,1.92E+02,1.92E+02,2.56E+02,1.92E+02,2.56E+02,3.20E+02]
loss_name = ['SL1','MSE','SL1','SL1','SL1','SL1','MSE','MSE','SL1','SL1','SL1','MSE','SL1','SL1','SL1','SL1','MSE','SL1','SL1','SL1']
init_name = ['KN','XU','XN','KU','KN','KU','KU','XU','XN','KU','XU','XN','XN','KN','XN','KU','KU','XU','KU','XU']
fan_mode = ['fan_out','','','fan_out','fan_in','fan_out','fan_in','','','fan_out','','','','fan_in','','fan_in','fan_in','','fan_in','']

# Scheduler
initial_lr = [8.00E-04,6.40E-03,6.40E-03,6.40E-03,2.00E-04,3.20E-03,3.20E-03,8.00E-04,1.28E-02,2.00E-04,1.28E-02,6.40E-03,8.00E-04,1.28E-02,1.28E-02,3.20E-03,4.00E-04,6.40E-03,3.20E-03,6.40E-03]
schd_min_lr = [5.00E-05,5.00E-05,5.00E-04,1.00E-04,5.00E-04,5.00E-05,1.00E-04,1.00E-04,5.00E-04,5.00E-05,2.00E-04,2.00E-04,1.00E-04,2.00E-04,5.00E-04,5.00E-04,2.00E-04,2.00E-04,2.00E-04,1.00E-04]

# Batch normalization
bn_momentum = [3.00E-01,4.00E-01,2.00E-01,1.00E-01,1.00E-01,7.00E-01,6.00E-01,1.00E-01,2.00E-01,3.00E-01,7.00E-01,7.00E-01,6.00E-01,2.00E-01,1.00E-01,4.00E-01,8.00E-01,1.00E-01,1.00E-01,8.00E-01]
act_after_bn = [True,True,True,True,True,False,True,False,False,False,False,False,False,True,True,True,False,False,True,True]
bn_a_trans = [True,True,False,False,False,False,True,False,True,False,True,False,False,True,True,True,False,True,True,True]

# Activation
act_fn_encode = [nn.LeakyReLU(negative_slope=.2),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.SELU(),nn.ReLU(),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.2),nn.ReLU(),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.2),nn.ReLU(),nn.LeakyReLU(negative_slope=.2),nn.SELU(),nn.ReLU(),nn.ReLU()]
act_fn_decode = [nn.LeakyReLU(negative_slope=.2),nn.ReLU(),nn.ReLU(),nn.ELU(),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.LeakyReLU(negative_slope=.2),nn.ELU(),nn.LeakyReLU(negative_slope=.2),nn.ELU(),nn.ReLU(),nn.ELU(),nn.LeakyReLU(negative_slope=.1),nn.ELU(),nn.LeakyReLU(negative_slope=.2),nn.LeakyReLU(negative_slope=.1),nn.ELU(),nn.LeakyReLU(negative_slope=.1)]
act_fn_output = [None,nn.LeakyReLU(negative_slope=.1),None,nn.LeakyReLU(negative_slope=.1),None,nn.LeakyReLU(negative_slope=.1),None,None,None,None,nn.LeakyReLU(negative_slope=.1),None,nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.1),None,nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.1),None,nn.LeakyReLU(negative_slope=.1)]
init_gain = ['leaky_relu','leaky_relu','leaky_relu','relu','relu','relu','selu','relu','leaky_relu','relu','leaky_relu','leaky_relu','relu','leaky_relu','leaky_relu','relu','leaky_relu','selu','relu','relu']
init_param = [.2,.1,.1,None,None,None,None,None,.1,None,.1,.2,None,.1,.2,None,.2,None,None,None]
act_a_trans = [True,True,False,True,True,False,False,True,True,False,False,False,True,False,False,True,False,True,True,False]

for duplicate in range(3):
    for iN, network_name in enumerate(network_names):
        if not (iN in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] and duplicate == 0):
            train(
                save_model_name=f'{network_names[iN]}{duplicate}',
                ngf=round(ngf[iN]),
                batch_size=round(bs_per_gpu[iN]*n_gpus),
                crop_size=round(crop_size[iN]),
                loss_name=loss_name[iN],
                init_name=init_name[iN],
                init_gain=init_gain[iN],
                init_param=init_param[iN],
                fan_mode=fan_mode[iN],
                initial_lr=initial_lr[iN],
                schd_min_lr=schd_min_lr[iN],
                bn_momentum=bn_momentum[iN],
                act_after_bn=act_after_bn[iN],
                bn_a_trans=bn_a_trans[iN],
                act_fn_encode=act_fn_encode[iN],
                act_fn_decode=act_fn_decode[iN],
                act_fn_output=act_fn_output[iN],
                act_a_trans=act_a_trans[iN],
            )