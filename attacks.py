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
from autoattack import AutoAttack
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

# Done
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

    return X_adv, X, robust_acc, natural_acc

# Done
def pgd_compose_RT(model, X_rtgs, Y, dataset_name, dataset, device_num, pgd_config, neptune_run, device):
    model.eval()
    if neptune_run:
        neptune_run['pgd_eval_config'] = pgd_config

    new_test_dataset = TensorDataset(X_rtgs, Y)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory= True)

    X_adv, X, robust_acc, natural_acc = pgd(model, dataset_name, dataset, new_test_dataloader, device_num, pgd_config, neptune_run, device)

    print("PGD Compose RT Grid Search Accuracy: ", float(robust_acc.detach().cpu()))
    print("RT Grid Search Accuracy: ", float(natural_acc.detach().cpu()))
    if neptune_run:
        neptune_run['pgd_compose_rt'] = float(robust_acc.detach().cpu())
        neptune_run['rt'] = float(natural_acc.detach().cpu())

# Done
def pgd_union_RT(model, X_rtgs, Y, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device):
    model.eval()
    if neptune_run:
        neptune_run['pgd_eval_config'] = pgd_config

    X_pgd, X, robust_acc, natural_acc = pgd(model, dataset_name, dataset, dataloader, device_num, pgd_config, neptune_run, device)
    print("PGD Accuracy: ", float(robust_acc.detach().cpu()))
    print("Natural Accuracy: ", float(natural_acc.detach().cpu()))
    if neptune_run:
        neptune_run['pgd'] = float(robust_acc.detach().cpu())
        neptune_run['natural'] = float(natural_acc.detach().cpu())

    with torch.no_grad():
        X_rtgs = X_rtgs.cuda(device_num)
        Y = Y.cuda(device_num)
        logits_rtgs = torch.zeros((X.shape[0], 10)).cuda(device_num)
        logits_pgd = torch.zeros((X.shape[0], 10)).cuda(device_num)
        for i in range(X.shape[0] // 128 + 1):
            start_idx = i * 128
            end_idx = (i+1) * 128

            logits_rtgs[start_idx:end_idx] = F.softmax(model(X_rtgs[start_idx:end_idx]), dim=1)
            logits_pgd[start_idx:end_idx] = F.softmax(model(X_pgd[start_idx:end_idx]), dim=1)

        preds_rtgs = torch.argmax(logits_rtgs, dim=1)
        preds_pgd = torch.argmax(logits_pgd, dim=1)

        X_final = torch.zeros_like(X)
        for i in range(len(X)):
            if preds_rtgs[i] != Y[i]:
                X_final[i] = X_rtgs[i]
            else:
               X_final[i] = X_pgd[i]

        X_final = X_final.detach().cpu()
        Y = Y.detach().cpu()
        final_dataset = TensorDataset(X_final, Y)
        final_dataloader = DataLoader(final_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory= True)

    model.eval()
    err_total = 0
    total = 0
    for data, target in final_dataloader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err, _, _ = _pgd_whitebox(model, X, y, epsilon=0, num_steps=1, step_size=0.01)
        err_total += err
        total += X.shape[0]

    acc = (total - err_total) / total

    print("PGD Union RT GS Accuracy: ", float(acc.detach().cpu()))
    if neptune_run:
        neptune_run['pgd_union_rt'] = float(acc.detach().cpu())

# Done
def aaa_compose_RT(model, X_rtgs, Y, dataset_name, dataset, device_num, aaa_config, neptune_run, device):
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

    # print("Running AA")
    # adversary = AutoAttack(model, norm='Linf', eps=aaa_config.ep, version='standard', device=device)
    # X_adv = adversary.run_standard_evaluation(X_rtgs, Y, bs=aaa_config.batch_size)

    # print(X_adv.shape)

    X_rtgs_numpy, Y_numpy = X_rtgs.permute(0,2,3,1).numpy(), Y.numpy()

    data_set = (
        X_rtgs_numpy,
        Y_numpy,
        aaa_config.dataset_name,
        aaa_config.dataset_name, #used for selecting AAA hyperparameters
    )
    
    model.eval()
    X_adv, robust_acc, broke_aaa = Adaptive_Auto_white_box_attack(model=model, 
                                   device=device, 
                                   eps=aaa_config.ep, 
                                   is_random=aaa_config.random, 
                                   batch_size=aaa_config.batch_size, 
                                   average_num=aaa_config.average_num, 
                                   model_name=None,#Only used for imagenet in AAA 
                                   data_set=data_set,
                                   Lnorm=aaa_config.Lnorm,
                                   neptune_run=neptune_run)
    
    if broke_aaa:
        print("AAA Compose RT GS Accuracy: ", 0)
        if neptune_run:
            neptune_run['aaa_compose_rt'] = 0.
    else:
        print("AAA Compose RT GS Accuracy: ", robust_acc)
        if neptune_run:
            neptune_run['aaa_compose_rt'] = robust_acc

    # with torch.no_grad():
    #     model.eval()
    #     X_adv = X_adv.cuda(device_num)
    #     Y = Y.cuda(device_num)
    #     logits_adv = torch.zeros((X_adv.shape[0], 10)).cuda(device_num)
    #     for i in range(X_adv.shape[0] // 128 + 1):
    #         start_idx = i * 128
    #         end_idx = (i+1) * 128

    #         logits_adv[start_idx:end_idx] = F.softmax(model(X_adv[start_idx:end_idx]), dim=1)

    #     preds_adv = torch.argmax(logits_adv, dim=1)
    #     acc = torch.mean((preds_adv == Y).float())
    #     print("AA Compose RT GS Accuracy (confirmed): ", float(acc.detach().cpu()))

# Done
def aaa_union_RT(model, X_rtgs, Y, dataset_name, dataset, dataloader, device_num, aaa_config, neptune_run, device):
    model.eval()
    if neptune_run:
        neptune_run['aaa_eval_config'] = aaa_config

    X_batches = []
    for data, _ in dataloader:
        data = data.to(device)
        X = Variable(data, requires_grad=True)
        X_batches.append(X)
    X = torch.cat(X_batches, dim=0).detach().cpu()

    X_numpy, Y_numpy = X.permute(0,2,3,1).numpy(), Y.numpy()

    data_set = (
        X_numpy,
        Y_numpy,
        aaa_config.dataset_name,
        aaa_config.dataset_name, #used for selecting AAA hyperparameters
    )
    
    model.eval()
    X_adv, robust_acc, broke_aaa = Adaptive_Auto_white_box_attack(model=model, 
                                   device=device, 
                                   eps=aaa_config.ep, 
                                   is_random=aaa_config.random, 
                                   batch_size=aaa_config.batch_size, 
                                   average_num=aaa_config.average_num, 
                                   model_name=None,#Only used for imagenet in AAA 
                                   data_set=data_set,
                                   Lnorm=aaa_config.Lnorm,
                                   neptune_run=neptune_run)
    
    if broke_aaa:
        print("AAA Accuracy: ", 0)
        print("AAA Union RT GS Accuracy: ", 0)
        if neptune_run:
            neptune_run['aaa'] = 0.
            neptune_run['aaa_union_rt'] = 0.

    else:
        print("AAA Accuracy: ", robust_acc)
        if neptune_run:
            neptune_run['aaa'] = robust_acc

    if not broke_aaa:
        with torch.no_grad():
            X_rtgs = X_rtgs.cuda(device_num)
            Y = Y.cuda(device_num)
            logits_rtgs = torch.zeros((X.shape[0], 10)).cuda(device_num)
            logits_aaa = torch.zeros((X.shape[0], 10)).cuda(device_num)
            for i in range(X.shape[0] // 128 + 1):
                start_idx = i * 128
                end_idx = (i+1) * 128

                logits_rtgs[start_idx:end_idx] = F.softmax(model(X_rtgs[start_idx:end_idx]), dim=1)
                logits_aaa[start_idx:end_idx] = F.softmax(model(X_adv[start_idx:end_idx]), dim=1)

            preds_rtgs = torch.argmax(logits_rtgs, dim=1)
            preds_aaa = torch.argmax(logits_aaa, dim=1)

            X_final = torch.zeros_like(X)
            for i in range(len(X)):
                if preds_rtgs[i] != Y[i]:
                    X_final[i] = X_rtgs[i]
                else:
                    X_final[i] = X_adv[i]

            X_final = X_final.detach().cpu()
            Y = Y.detach().cpu()
            final_dataset = TensorDataset(X_final, Y)
            final_dataloader = DataLoader(final_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory= True)

        model.eval()
        err_total = 0
        total = 0
        for data, target in final_dataloader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err, _, _ = _pgd_whitebox(model, X, y, epsilon=0, num_steps=1, step_size=0.01)
            err_total += err
            total += X.shape[0]

        acc = (total - err_total) / total

        print("AAA Union RT GS Accuracy: ", float(acc.detach().cpu()))
        if neptune_run:
            neptune_run['aaa_union_rt'] = float(acc.detach().cpu())