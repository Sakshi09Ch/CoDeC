#Modified from https://github.com/LYang-666/TRGP/tree/main/dataloader

from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import transforms

class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train, transform=None):
        super(MiniImageNet, self).__init__()
        self.transform = transform
        if train:
            self.name='train'
        else:
            self.name='test'
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['images']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
