from deploy import deploy
import torch.nn as nn


# Parameters
n_gpus = 4

# Networks
network_names = ['D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W']

# General
ngf = [9.60E+01,1.28E+02,3.20E+01,6.40E+01,9.60E+01,6.40E+01,1.28E+02,9.60E+01,3.20E+01,1.60E+02,6.40E+01,3.20E+01,1.28E+02,9.60E+01,6.40E+01,3.20E+01,6.40E+01,1.28E+02,1.60E+02,6.40E+01]
bs_per_gpu = [1.60E+01,4.00E+00,1.60E+01,4.00E+00,2.00E+00,1.60E+01,2.00E+00,4.00E+00,2.00E+00,2.00E+00,2.00E+00,1.60E+01,2.00E+00,2.00E+00,4.00E+00,8.00E+00,2.00E+00,2.00E+00,1.60E+01,1.60E+01]
crop_size = [1.28E+02,2.24E+02,3.20E+02,3.52E+02,1.28E+02,6.40E+01,2.56E+02,6.40E+01,2.56E+02,1.92E+02,1.92E+02,6.40E+01,3.20E+02,2.56E+02,6.40E+01,1.92E+02,2.56E+02,2.88E+02,6.40E+01,1.28E+02]
clip_gradients = [False,True,False,True,False,True,False,True,True,True,True,False,True,True,False,False,False,True,True,False]
optim_name = ['Adam','SGD','Adam','SGD','Adam','SGD','Adam','Adam','SGD','Adam','Adam','Adam','Adam','SGD','Adam','Adam','Adam','Adam','Adam','SGD']
loss_name = ['SL1','MSE','SL1','L1','SL1','MSE','MSE','L1','SL1','L1','MSE','MSE','SL1','SL1','L1','SL1','MSE','SL1','SL1','SL1']
init_name = ['XU','XU','XN','XU','KN','XU','KU','KU','KN','XN','XU','XN','KN','XU','XU','KU','KU','KU','KN','XN']
init_gain = ['relu','selu','leaky_relu','relu','relu','relu','selu','relu','selu','leaky_relu','leaky_relu','relu','leaky_relu','relu','leaky_relu','relu','leaky_relu','leaky_relu','selu','leaky_relu']
init_param = [None,None,.1,None,None,None,None,None,None,.1,.01,None,.1,None,None,None,.2,.01,None,None]
fan_mode = ['fan_in','fan_out','fan_out','fan_out','fan_in','fan_out','fan_in','fan_in','fan_out','fan_in','fan_out','fan_out','fan_in','fan_out','fan_in','fan_in','fan_in','fan_out','fan_out','fan_in']

# Regularization
spat_drop_p = [2.00E-02,3.00E-01,2.00E-01,5.00E-01,1.00E-01,3.00E-01,1.00E-02,2.50E-01,1.50E-01,2.00E-01,5.00E-02,3.00E-01,1.00E-02,2.50E-01,2.00E-02,4.50E-01,3.00E-01,1.00E-02,2.00E-01,2.50E-01]
noise = [3.00E-01,2.50E-01,5.00E-01,2.00E-02,1.00E-01,1.50E-01,1.00E-01,1.00E-02,2.00E-02,2.00E-01,4.00E-01,1.50E-01,3.00E-01,1.00E-01,5.00E-01,5.00E-02,3.50E-01,2.50E-01,4.00E-01,5.00E-02]
localdeform = [2.00E+00,9.00E+00,8.00E+00,3.00E+00,1.00E+01,4.00E+00,1.20E+01,6.00E+00,4.00E+00,3.00E+00,1.20E+01,2.00E+00,1.00E+01,6.00E+00,7.00E+00,3.00E+00,2.00E+00,1.00E+00,6.00E+00,4.00E+00]
weight_decay = [5.00E-04,1.00E-05,5.00E-04,1.00E-03,1.00E-05,1.00E-04,5.00E-04,1.00E-03,5.00E-05,5.00E-05,1.00E-04,1.00E-05,1.00E-04,5.00E-04,5.00E-05,1.00E-05,1.00E-05,1.00E-05,5.00E-04,5.00E-05]

# Scheduler
initial_lr = [2.56E-02,3.20E-03,6.40E-03,1.28E-02,2.00E-04,3.20E-03,3.20E-03,2.56E-02,6.40E-03,3.20E-03,4.00E-04,2.56E-02,2.56E-02,1.60E-03,4.00E-04,3.20E-03,4.00E-04,2.56E-02,2.00E-04,2.00E-04]
schd_min_lr = [1.00E-04,2.00E-04,5.00E-04,5.00E-04,5.00E-04,1.00E-04,1.00E-04,5.00E-04,5.00E-04,5.00E-04,5.00E-04,1.00E-04,2.00E-04,2.00E-04,5.00E-04,5.00E-04,2.00E-04,1.00E-04,5.00E-04,5.00E-04]

# Batch normalization
bn = [True,True,True,True,True,True,True,False,True,False,False,False,True,False,False,True,True,True,False,True]
bn_momentum = [1.00E-01,3.00E-01,2.00E-01,3.00E-01,1.00E-01,3.00E-01,6.00E-01,1.00E-01,2.00E-01,1.00E-01,1.00E-01,1.00E-01,5.00E-01,1.00E-01,1.00E-01,4.00E-01,8.00E-01,1.00E-01,1.00E-01,3.00E-01]
act_after_bn = [False,False,True,False,True,False,True,False,False,True,True,True,True,False,False,True,False,True,False,True]
bn_a_trans = [True,False,False,False,False,True,True,True,False,False,True,True,True,False,False,True,False,False,True,False]

# Activation
act_fn_encode = [nn.ReLU(),nn.SELU(),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.SELU(),nn.ReLU(),nn.SELU(),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.01),nn.ReLU(),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.ELU(),nn.ReLU(),nn.LeakyReLU(negative_slope=.2),nn.LeakyReLU(negative_slope=.01),nn.SELU(),nn.ELU()]
act_fn_decode = [nn.ReLU(),nn.SELU(),nn.ReLU(),nn.ELU(),nn.LeakyReLU(negative_slope=.1),nn.LeakyReLU(negative_slope=.1),nn.ReLU(),nn.ReLU(),nn.LeakyReLU(negative_slope=.01),nn.ReLU(),nn.ELU(),nn.SELU(),nn.LeakyReLU(negative_slope=.2),nn.SELU(),nn.ELU(),nn.ELU(),nn.LeakyReLU(negative_slope=.2),nn.LeakyReLU(negative_slope=.01),nn.ReLU(),nn.ELU()]
act_fn_output = [nn.ReLU(),nn.Sigmoid(),None,nn.Tanh(),None,None,None,nn.LeakyReLU(negative_slope=.1),None,nn.ReLU(),None,nn.ReLU(),nn.Sigmoid(),None,nn.Tanh(),None,nn.LeakyReLU(negative_slope=.1),nn.Tanh(),nn.LeakyReLU(negative_slope=.1),nn.ReLU()]
act_a_trans = [True,False,False,True,True,False,False,True,False,True,False,False,True,True,True,True,False,True,True,True]

for duplicate in range(3):
    for iN, network_name in enumerate(network_names):
        deploy(
            load_model_name=f'{network_names[iN]}{duplicate}',
            ngf=round(ngf[iN]),
            spat_drop_p=spat_drop_p[iN],       
            act_fn_encode=act_fn_encode[iN],
            act_fn_decode=act_fn_decode[iN],
            act_fn_output=act_fn_output[iN],
            init_name=init_name[iN],
            init_gain=init_gain[iN],
            init_param=init_param[iN],
            fan_mode=fan_mode[iN],
            act_a_trans=act_a_trans[iN],
            bn_a_trans=bn_a_trans[iN],
            bn=bn[iN],
            act_after_bn=act_after_bn[iN],
            bn_momentum=bn_momentum[iN],
            crop_size=round(crop_size[iN]),
            batch_size=round(bs_per_gpu[iN]*n_gpus),
        )