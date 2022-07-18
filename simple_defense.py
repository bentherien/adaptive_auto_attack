from __future__ import print_function
import os
import sys
import argparse
import torch
import torchvision
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import neptune.new as neptune

sys.path.append("./TRADES")

from mmcv import Config
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from models.net_mnist import *
from models.small_cnn import *
from new_trades_losses import (trades_loss_ORIG, trades_loss_linfty_compose_RT, 
                               trades_loss_linfty_u_RT, trades_loss_RT,)
# from attacks import aaa_compose_w10, pgd_compose_w10, aaa_union_w10, pgd_union_w10, pgd, aaa, w10, natural_accuracy
from attacks import pgd, pgd_compose_w10, pgd_union_w10

def get_model(model_name,device,verbose=False):
    if model_name == 'SmallCNN':
        model = SmallCNN().to(device)
    elif model_name == 'WRN-34-10':
        model = WideResNet().to(device)

    if verbose:
        from torchsummary import summary
        s = summary(model,trainset[0][0].shape)
    return model