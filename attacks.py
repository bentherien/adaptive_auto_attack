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
from torch.autograd import Variable
from augmentation_tools import get_w10_images, get_gridsearch_images, show_reg_aug_side_by_side_numpy
from new_trades_losses import (trades_loss_ORIG, trades_loss_linfty_compose_RT, 
                               trades_loss_linfty_u_RT, trades_loss_RT,)
from proj_Adaptive_Auto_Attack_main_PROJECT import Adaptive_Auto_white_box_attack
from torch.utils.data import TensorDataset, DataLoader

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd, X_pgd

def pgd(model, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device):
    """
    evaluate model by white-box attack
    """

    if neptune_run:
        neptune_run['pgd_eval_config'] = pgd_config

    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    total = 0

    X_adv_batches = []
    X_batches = []
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, X_adv_batch = _pgd_whitebox(model, X, y, epsilon=pgd_config.pgd_epsilon, num_steps=pgd_config.pgd_num_steps, step_size=pgd_config.pgd_step_size)
        robust_err_total += err_robust
        natural_err_total += err_natural
        total += X.shape[0]
        X_adv_batches.append(X_adv_batch)
        X_batches.append(X)

    robust_acc = (total - robust_err_total) / total 
    natural_acc = (total - natural_err_total) / total
    X_adv = torch.cat(X_adv_batches, dim=0)
    X = torch.cat(X_batches, dim=0)
    print(X_adv.shape, X.shape)

    if neptune_run:
        neptune_run['pgd_robust_acc'] = robust_acc 
        neptune_run['pgd_natural_acc'] = natural_acc
    
    print('robust acc: ', robust_acc)
    print('natural acc: ', natural_acc)

    return X_adv, X

def pgd_compose_w10(model, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device):
    if neptune_run:
        neptune_run['pgd_eval_config'] = pgd_config

    w10_X, w10_Y = get_w10_images(dataset,dataloader,model,device_num)
    w10_X, w10_Y = w10_X.detach().cpu(), w10_Y.detach().cpu()

    new_test_dataset = TensorDataset(w10_X, w10_Y)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory= True)

    X_adv, X = pgd(model, dataset_name, dataset, new_test_dataloader, device_num, pgd_config, neptune_run, device)

def pgd_union_w10(model, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device):
    if neptune_run:
        neptune_run['pgd_eval_config'] = pgd_config

    print("Constructing RT adversarial examples")
    w10_X, w10_Y = get_w10_images(dataset,dataloader,model,device_num)
    print("Constructing L_infty PGD adversarial examples")
    pgd_X, X = pgd(model, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device)

    with torch.no_grad():
        print("Evaluating union")
        losses = torch.zeros((X.shape[0], 2)).cuda(device_num)
        criterion = nn.KLDivLoss(reduce=False)
        for i in range(X.shape[0] // 128 + 1):
            start_idx = i * 128
            end_idx = (i+1) * 128

            losses[start_idx:end_idx, 0] = torch.sum(criterion(F.log_softmax(model(w10_X[start_idx:end_idx]), dim=1),
                                                                F.softmax(model(X[start_idx:end_idx]), dim=1)), dim=1)

            losses[start_idx:end_idx, 1] = torch.sum(criterion(F.log_softmax(model(pgd_X[start_idx:end_idx]), dim=1),
                                                                F.softmax(model(X[start_idx:end_idx]), dim=1)), dim=1)

            if end_idx > 10000:
                print(end_idx)

        print("Constructing RT union L_infty PGD adversarial examples")
        X_final = torch.zeros_like(X)
        union_index = torch.argmax(losses, dim=1)
        for i in range(len(X)):
            if union_index[i] == 0:
                X_final[i] = w10_X[i]
            else:
                X_final[i] = pgd_X[i]

        print("Evaluating on adversarial examples")
        X_final = X_final.detach().cpu()
        w10_Y = w10_Y.detach().cpu()
        final_dataset = TensorDataset(X_final, w10_Y)
        final_dataloader = DataLoader(final_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory= True)

    model.eval()
    err_total = 0
    total = 0

    for data, target in final_dataloader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err, _, X_adv_batch = _pgd_whitebox(model, X, y, epsilon=0, num_steps=1, step_size=0.01)
        err_total += err
        total += X.shape[0]

    acc = (total - err_total) / total

    print("Union accuracy", acc)

def aaa_compose_w10(model, dataset_name, dataset, dataloader, device_num, aaa_config, neptune_run, device):
    """Function tests the current model using the AAA attack.

    Args:
        model (torch.nn.Module): the model tested
        dataset_name (string): the name of the dataset
        dataset (torch.util.dataset): the dataset
        dataloader (torch.util.dataloader): the test dataloader
        device_num (int): the gpu number to use
        aaa_config (config object): the config object for aaa attack
        neptune_run (neptune object): logger for neptune.ai 
    """
    if neptune_run:
        neptune_run['aaa_eval_config'] = aaa_config

    w10_X, w10_Y = get_w10_images(dataset,dataloader,model,device_num)
    print(w10_X.shape,w10_Y.shape)
    
    print(w10_X.permute(0,2,3,1).shape,w10_Y.shape)
    w10_X, w10_Y = w10_X.permute(0,2,3,1).detach().cpu().numpy(), w10_Y.detach().cpu().numpy()

    data_set = (
        w10_X,
        w10_Y,
        aaa_config.dataset_name,
        aaa_config.dataset_name, #used for selecting AAA hyperparameters
    )
    
    Adaptive_Auto_white_box_attack(model=model, 
                                   device=device, 
                                   eps=aaa_config.ep, 
                                   is_random=aaa_config.random, 
                                   batch_size=aaa_config.batch_size, 
                                   average_num=aaa_config.average_num, 
                                   model_name=None,#Only used for imagenet in AAA 
                                   data_set=data_set,
                                   Lnorm=aaa_config.Lnorm,
                                   neptune_run=neptune_run)


def aaa_compose_gridsearch(model, dataset_name, dataset, dataloader, device_num, aaa_config, neptune_run, device):
    """Function tests the current model using the AAA attack.

    Args:
        model (torch.nn.Module): the model tested
        dataset_name (string): the name of the dataset
        dataset (torch.util.dataset): the dataset
        dataloader (torch.util.dataloader): the test dataloader
        device_num (int): the gpu number to use
        aaa_config (config object): the config object for aaa attack
        neptune_run (neptune object): logger for neptune.ai 
    """
    if neptune_run:
        neptune_run['aaa_eval_config'] = aaa_config

    gridsearch_X, gridsearch_Y = get_gridsearch_images(dataset,dataloader,model,device_num)
    print(gridsearch_X.shape,gridsearch_Y.shape)

    # cmap = 'gray' if dataset_name == "MNIST" else None
    # if neptune_run:
    #     show_reg_aug_side_by_side_numpy(dataset_regular=dataset,
    #                                     images_aug=w10_X,
    #                                     labels=w10_Y,
    #                                     classes=dataset.classes,
    #                                     total_plots=40,
    #                                     plots_per_row=5,
    #                                     figsize=(20,67),
    #                                     savepath='/tmp/aaa_test_images.png',
    #                                     cmap=cmap)
    #     neptune_run["aaa_compose_w10/side_by_side"].upload('/tmp/aaa_test_images.png')
    
    print(gridsearch_X.permute(0,2,3,1).shape,gridsearch_Y.shape)
    gridsearch_X, gridsearch_Y = gridsearch_X.permute(0,2,3,1).detach().cpu().numpy(), gridsearch_Y.detach().cpu().numpy()

    data_set = (
        gridsearch_X,
        gridsearch_Y,
        aaa_config.dataset_name,
        aaa_config.dataset_name, #used for selecting AAA hyperparameters
    )

    
    Adaptive_Auto_white_box_attack(model=model, 
                                   device=device, 
                                   eps=aaa_config.ep, 
                                   is_random=aaa_config.random, 
                                   batch_size=aaa_config.batch_size, 
                                   average_num=aaa_config.average_num, 
                                   model_name=None,#Only used for imagenet in AAA 
                                   data_set=data_set,
                                   Lnorm=aaa_config.Lnorm,
                                   neptune_run=neptune_run)