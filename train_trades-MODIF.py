from __future__ import print_function
import os
import sys
import argparse
import torch
import torchvision

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
from new_trades_losses import trades_loss_ORIG, trades_loss_linfty_compose_RT, trades_loss_linfty_u_RT, trades_loss_RT

parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument('--batch-size', '-bs', type=int, #default=128, 
                    metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, #default=128, 
                    metavar='N',help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, #default=76, 
                    metavar='N',help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', #default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, #default=0.1, 
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, #default=0.9, 
                    metavar='M',help='SGD momentum')
parser.add_argument('--epsilon', #default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', #default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', #default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', #default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, #default=1, 
                    metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, #default=100, 
                    metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', #default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', #default=1, 
                    type=int, metavar='N',help='save frequency')
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

#override via CLI
cfg._cfg_dict.update({k:v for k,v in args.__dict__.items() if v != None})

if args.no_neptune:
    neptune_run = None
else:
    cfg_key = Config.fromfile('./keys/api_key.py')
    neptune_run = neptune.init(**cfg_key.neptune_args) 
    neptune_run['config'] = cfg
    neptune_run['cli_args'] = args

if cfg.trades_loss == 'orig':
    trades_loss = trades_loss_ORIG
elif cfg.trades_loss == 'linfty_compose_RT':
    trades_loss = trades_loss_linfty_compose_RT
elif cfg.trades_loss == 'linfty_u_RT':
    trades_loss = trades_loss_linfty_u_RT
elif cfg.trades_loss == 'RT':
    trades_loss = trades_loss_RT


print(cfg)

# settings
model_dir = cfg.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not cfg.no_cuda and torch.cuda.is_available()
torch.manual_seed(cfg.seed)
device = torch.device("cuda:{}".format(cfg.device_num) if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}





if cfg.dataset == 'cifar10':
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, **kwargs)
elif cfg.dataset == 'MNIST':
    trainset = datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=cfg.batch_size,shuffle=True, **kwargs)
    testset = datasets.MNIST('../data',train=False,transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset,batch_size=cfg.test_batch_size, shuffle=False, **kwargs)
else:
    raise NotImplementedError("Invalid dataset name: {}".format(cfg.dataset))



def train(cfg, model, device, train_loader, optimizer, epoch, device_num, neptune_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           neptune_run=neptune_run,
                           step_size=cfg.step_size,
                           epsilon=cfg.epsilon,
                           perturb_steps=cfg.num_steps,
                           beta=cfg.beta,
                           device_num=device_num)

        # print(loss)
        loss.backward()
        optimizer.step()

        if neptune_run:
            neptune_run['trades_loss'].log(loss.item())

        # print progress
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            # print(data,target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate_cifar10(optimizer, epoch):
    """decrease the learning rate"""
    lr = cfg.lr
    if epoch >= 75:
        lr = cfg.lr * 0.1
    if epoch >= 90:
        lr = cfg.lr * 0.01
    if epoch >= 100:
        lr = cfg.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_mnist(optimizer, epoch):
    """decrease the learning rate"""
    lr = cfg.lr
    if epoch >= 55:
        lr = cfg.lr * 0.1
    if epoch >= 75:
        lr = cfg.lr * 0.01
    if epoch >= 90:
        lr = cfg.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model(model_name,device,verbose=False):
    if model_name == 'SmallCNN':
        model = SmallCNN().to(device)
    elif model_name == 'WRN-34-10':
        model = WideResNet().to(device)

    if verbose:
        from torchsummary import summary
        s = summary(model,trainset[0][0].shape)
    return model


def main():
    # init model, ResNet18() can be also used here for training
    model = get_model(cfg.model_name,device,verbose=cfg.verbose)
  
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        # adjust learning rate for SGD
        if cfg.dataset == 'MNIST':
            adjust_learning_rate_mnist(optimizer, epoch)
        elif cfg.dataset == 'cifar10':
            adjust_learning_rate_cifar10(optimizer, epoch)

        # adversarial training
        train(cfg, model, device, train_loader, optimizer, epoch, cfg.device_num, neptune_run)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % cfg.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-{}-epoch{}.pt'.format(cfg.model_name,epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}-checkpoint_epoch{}.tar'.format(cfg.model_name,epoch)))


if __name__ == '__main__':
    main()
