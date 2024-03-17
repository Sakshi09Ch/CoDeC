import torch

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

from torch.distributions.uniform import Uniform


__all__ = ['alexnet', 'get_representation_matrix']

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet_Scaled(nn.Module):
    def __init__(self,taskcla, factor):
        super(AlexNet_Scaled, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, int(64/factor), 4, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64/factor), track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(int(64/factor), int(128/factor), 3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(128/factor), track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(int(64/factor))
        self.map.append(s)
        self.conv3 = nn.Conv2d(int(128/factor), int(256/factor), 2, bias=False)
        self.bn3 = nn.BatchNorm2d(int(256/factor), track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(int(128/factor))
        self.map.append(int(256/factor)*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(int(256/factor)*self.smid*self.smid,int(2048/factor), bias=False)
        self.bn4 = nn.BatchNorm1d(int(2048/factor), track_running_stats=False)
        self.fc2 = nn.Linear(int(2048/factor),int(2048/factor), bias=False)
        self.bn5 = nn.BatchNorm1d(int(2048/factor), track_running_stats=False)
        self.map.extend([int(2048/factor)])
        
        self.taskcla = taskcla

        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(int(2048/factor),n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y=[]

        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y



def alexnet_scaled(taskcla):
    return AlexNet_Scaled(taskcla, 4)  
