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
from augmentation_tools import get_gridsearch_images
from attacks import pgd, pgd_compose_RT, pgd_union_RT, aaa_compose_RT, aaa_union_RT

def get_model(model_name,device,verbose=False):
    if model_name == 'SmallCNN':
        model = SmallCNN().to(device)
    elif model_name == 'WRN-34-10':
        model = WideResNet().to(device)

    if verbose:
        from torchsummary import summary
        s = summary(model,trainset[0][0].shape)
    return model

parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
#REQUIRED OVERRIDE TO CHANGE                    
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-neptune', '-nn', default=False, action='store_true', 
                    help='Whether to use neptune logging or not')
parser.add_argument('--config', '-c', default='./config/defenses/default_runtime.py', 
                    type=str, metavar='CFG', help='Whether to use neptune logging or not')
parser.add_argument('--device-num', '-dn', default=0, 
                    type=int, required=True, help='The number of the GPU to use')


args = parser.parse_args()
cfg = Config.fromfile(args.config)

cfg._cfg_dict.update({k:v for k,v in args.__dict__.items() if v != None})

print(cfg)

identifier = "{}_{}_{}_{}".format(cfg.dataset, cfg.trades_loss, cfg.seed, cfg.beta)
print("Model identifier: ", identifier)

if args.no_neptune:
    neptune_run = None
else:
    cfg_key = Config.fromfile('./keys/api_key.py')
    
    neptune_run = neptune.init(**cfg_key.neptune_args) 
    neptune_run['config'] = cfg
    neptune_run['name'] = identifier
    neptune_run['cli_args'] = args

# settings
model_dir = os.path.join(cfg.model_dir, identifier)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not cfg.no_cuda and torch.cuda.is_available()
torch.manual_seed(cfg.seed)
device = torch.device("cuda:{}".format(cfg.device_num) if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if cfg.dataset == 'cifar10':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, **kwargs)
elif cfg.dataset == 'MNIST':
    testset = datasets.MNIST('../data',train=False,transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset,batch_size=cfg.test_batch_size, shuffle=False, **kwargs)
else:
    raise NotImplementedError("Invalid dataset name: {}".format(cfg.dataset))


model = get_model(cfg.model_name, device, verbose=cfg.verbose)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model-{}-final.pt'.format(cfg.model_name)), map_location=device))

print("Evaluating on attack suite.")

# First get worst-case RT images
X_rtgs, Y = get_gridsearch_images(testset, test_loader, model, cfg.device_num)
X_rtgs = X_rtgs.detach().cpu()
Y = Y.detach().cpu()

pgd_compose_RT(
        model=model,
        X_rtgs=X_rtgs,
        Y=Y,
        dataset_name=cfg.dataset,
        dataset=testset,
        device_num=cfg.device_num,
        pgd_config=Config.fromfile(cfg.pgd_config),
        neptune_run=neptune_run,
        device=device
)

pgd_union_RT(
        model=model,
        X_rtgs=X_rtgs,
        Y=Y,
        dataset_name=cfg.dataset,
        dataset=testset,
        dataloader=test_loader,
        device_num=cfg.device_num,
        pgd_config=Config.fromfile(cfg.pgd_config),
        neptune_run=neptune_run,
        device=device
)

aaa_union_RT(
        model=model,
        X_rtgs=X_rtgs,
        Y=Y,
        dataset_name=cfg.dataset,
        dataset=testset,
        dataloader=test_loader,
        device_num=cfg.device_num,
        aaa_config=Config.fromfile(cfg.aaa_config),
        neptune_run=neptune_run,
        device=device
)

aaa_compose_RT(
        model=model,
        X_rtgs=X_rtgs,
        Y=Y,
        dataset_name=cfg.dataset,
        dataset=testset,
        device_num=cfg.device_num,
        aaa_config=Config.fromfile(cfg.aaa_config),
        neptune_run=neptune_run,
        device=device
)
