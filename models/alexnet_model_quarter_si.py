import torch

#from torchinfo import summary

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import numpy as np
import random
import pdb
import argparse,time
import math
import copy
from copy import deepcopy

from trainer_gatherSI_alexnet import *
from gossip_choco import *
from torch.distributions.uniform import Uniform


__all__ = ['alexnet_scaled_si', 'get_representation_matrix']

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet_Scaled_SI(nn.Module):
    def __init__(self, model, n_outputs, n_tasks, args):
        super(AlexNet_Scaled_SI, self).__init__()
        
        self.net = model
        self.net_old= None
        self.fisher = None
        self.n_outputs = n_outputs
        self.nc_per_task = n_outputs / n_tasks
        self.lamb = args.ewc_lamb # define ewc (\lambda) or SI regularization (c)  
        self.n_tasks = n_tasks
        self.args = args

        self.current_task = 0
        self.loss = torch.nn.CrossEntropyLoss()

        self.epoch = 0
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances
        self.pass_itr = 0
        self.real_epoch = 0

    def initialize_optimizer(self):
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)
        print (">> Reset optimizer with lr : ",self.args.lr)
        # pass

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        print (">> Initializing Buffers for SI computation : only once <<")
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self, rank, t):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        if(rank==0):
            print (">> Reset W and p_old once for each task <<")
        W = {}
        p_old = {}
        # for gen_params in self.param_list:
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old, rank):
        '''Update the running parameter importance estimates in W.'''
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    W[n].add_(-p.grad*(p.detach()-p_old[n]))
                p_old[n] = p.detach().clone()
    
        return W, p_old

    def update_omega(self, W, epsilon, rank):
        '''After completing training on a context, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        omega_send= {}
        p_send={}
        print (">> Update Omega - Once AFTER each task <<")
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add
                omega_send.update({n: omega_new})
                p_send.update({n: p_current})

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

        return omega_send, p_send
                
    def store_omega(self, omega):
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_SI_omega'.format(n), omega[n])

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.net.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def observe(self, model, x, y, t, W, p_old, epoch, iterate, rank):
        losses = AverageMeter()
        top1 = AverageMeter()

        self.net.train() 
        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.net.zero_grad()
            self.opt.zero_grad()

            if t != self.current_task:
                self.current_task = t

            logits  = self.net.forward(x)[t]

            loss_ce = self.loss(logits, y)

            # SI - Regularization loss 
            loss_reg=0
            if t>0:   loss_reg = self.surrogate_loss()
            # total Loss         
            loss = loss_ce + self.lamb * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip_norm)
            self.opt.step()

            res = model.transfer_params(epoch=epoch+(1e-3*iterate), lr=args.lr)    

            logits = logits.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(logits.data, y)[0]
            losses.update(loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))

            if iterate % args.print_freq == 0:
                print('Rank: {0}\t'
                  'Epoch: [{1}][{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, iterate, loss=losses, top1=top1)) 


        return loss.item(), W, p_old

def alexnet_scaled_si(model, n_outputs, n_tasks, args):
    return AlexNet_Scaled_SI(model, n_outputs, n_tasks, args)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
