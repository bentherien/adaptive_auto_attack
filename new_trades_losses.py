
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim

from torch.autograd import Variable

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades_loss_ORIG(model,
                    x_natural,
                    y,
                    optimizer,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=1.0,
                    distance='l_inf',
                    device_num=0,
                    neptune_run=None):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(device_num).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda(device_num).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss_robust = beta * loss_robust
    if neptune_run:
        neptune_run['training_robust_loss'].log(loss_robust.item())
        neptune_run['training_natural_loss'].log(loss_natural.item())
    loss = loss_natural + loss_robust
    return loss


def select_WX(model, x_natural, y, criterion, device_num):
    """
    Given a batch of inputs, this method selects the worst of 
    10 uniformly sampled spatial transformations.

    Args: 
        model (nn.Module): the model being trained 
        x_natural (torch.Tensor): the current input batch 

    Returns: 
        the batch of spatially transformed images which maximize loss
    """
    with torch.no_grad():
        # Shape: [10, bs, channels, width, height]
        affine_x_10 = torch.zeros((10,) + x_natural.shape).cuda(device_num)
        losses = torch.zeros(10, x_natural.shape[0]).cuda(device_num)
        affine_T = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(1, 1), interpolation=T.InterpolationMode.BILINEAR)
        for i in range(10):
            # Apply random affine transformation to each image in the batch
            affine_x = torch.cat([affine_T(x_natural[j]).unsqueeze(0) for j in range(len(x_natural))], dim=0)
            affine_x_10[i] = affine_x

            losses[i] = torch.sum(criterion(F.log_softmax(model(affine_x), dim=1),
                            F.softmax(model(x_natural), dim=1)), dim=1)

        # Identify, for each image, the transform (of the 10) that maximizes the loss
        wc10_index = torch.argmax(losses, dim=0)

        # Should use torch gather here
        affine_x_10 = torch.transpose(affine_x_10, 0, 1)
        wc10_x = torch.zeros_like(x_natural)
        for i in range(len(x_natural)):
            wc10_x[i] = affine_x_10[i][wc10_index[i]]

    return wc10_x

def trades_loss_linfty_u_RT(model,
                            x_natural,
                            y,
                            optimizer,
                            step_size=0.003,
                            epsilon=0.031,
                            perturb_steps=10,
                            beta=1.0,
                            distance='l_inf',
                            device_num=0,
                            neptune_run=None):
    
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_kl_SST = nn.KLDivLoss(reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    ### MODIFICATION START

    x_spatial_adv = select_WX(model, x_natural, y, criterion_kl_SST, device_num)

    ### MODIFICATION_END

    x_linfty_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(device_num).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_linfty_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_linfty_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_linfty_adv])[0]
            x_linfty_adv = x_linfty_adv.detach() + step_size * torch.sign(grad.detach())
            x_linfty_adv = torch.min(torch.max(x_linfty_adv, x_natural - epsilon), x_natural + epsilon)
            x_linfty_adv = torch.clamp(x_linfty_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda(device_num).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_linfty_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_linfty_adv = torch.clamp(x_linfty_adv, 0.0, 1.0)
    model.train()

    x_linfty_adv = Variable(torch.clamp(x_linfty_adv, 0.0, 1.0), requires_grad=False)
    x_spatial_adv = Variable(torch.clamp(x_spatial_adv, 0.0, 1.0), requires_grad=False)
    
    ### MODIFICATION START
    # Here we construct x_adv

    # index 0 corresponds to linfty perturbation and index 1 corresponds to spatial perturbation
    losses = torch.zeros(2, x_linfty_adv.shape[0]).cuda(device_num)

    losses[0] = torch.sum(criterion(F.log_softmax(model(x_linfty_adv), dim=1),
                            F.softmax(model(x_natural), dim=1)), dim=1)
    losses[1] = torch.sum(criterion(F.log_softmax(model(x_spatial_adv), dim=1),
                            F.softmax(model(x_natural), dim=1)), dim=1)

    best_perturbation_index = torch.argmax(losses, dim=0)

    # Should use torch gather here
    affine_x_10 = torch.transpose(affine_x_10, 0, 1)
    x_adv = torch.zeros_like(x_natural)
    for i in range(len(x_natural)):
        if best_perturbation_index[i] == 0:
            x_adv[i] = x_linfty_adv[i]
        else:
            x_adv[i] = x_spatial_adv[i]

    if neptune_run:
        # proportion of adversarial images selected by max strategy that are l_infty perturbations
        proportion_linfty = torch.mean(best_perturbation_index == 0)
        neptune_run['proportion_linfty'].log(proportion_linfty)

    ### MODIFICATION END    
    
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def trades_loss_RT(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                device_num=0,
                neptune_run=None):
    
    # define KL-loss
    criterion_kl_SST = nn.KLDivLoss(reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    ### MODIFICATION START

    x_adv = select_WX(model, x_natural, y, criterion_kl_SST, device_num)

    ### MODIFICATION_END
    
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def trades_loss_linfty_compose_RT(model,
                        x_natural,
                        y,
                        optimizer,
                        step_size=0.003,
                        epsilon=0.031,
                        perturb_steps=10,
                        beta=1.0,
                        distance='l_inf',
                        device_num=0,
                        neptune_run=None):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_kl_SST = nn.KLDivLoss(reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    ### MODIFICATION START

    x_adv_start = select_WX(model, x_natural, y, criterion_kl_SST, device_num)

    ### MODIFICATION_END

    x_adv = x_adv_start.detach() + 0.001 * torch.randn(x_adv_start.shape).cuda(device_num).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_adv_start), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_adv_start - epsilon), x_adv_start + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_adv_start.shape).cuda(device_num).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_adv_start + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_adv_start), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_adv_start)
            delta.data.clamp_(0, 1).sub_(x_adv_start)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_adv_start + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss_robust = beta * loss_robust
    if neptune_run:
        neptune_run['training_robust_loss'].log(loss_robust.item())
        neptune_run['training_natural_loss'].log(loss_natural.item())
    loss = loss_natural + loss_robust
    return loss