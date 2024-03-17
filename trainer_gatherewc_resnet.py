##EWC implementation modified from https://github.com/joansj/hat/blob/6fd0b9d98e089bec1f9da6cc20550916ce196829/src/approaches/ewc.py#L98
import argparse
import os
import shutil
import time
import numpy as np
import statistics 
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from math import ceil
from copy import deepcopy
from random import Random

# Importing modules related to distributed processing
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn
###########
from gossip_choco import GossipDataParallel
from gossip_choco import RingGraph, GridGraph
from gossip_choco import UniformMixing
from gossip_choco import *
from models import *

import notmnist_setup
import miniimagenet_setup

import medmnist
from medmnist import INFO, Evaluator


parser = argparse.ArgumentParser(description='Propert AlexNet for CIFAR10/CIFAR100 in pytorch')
parser.add_argument('--devices', default=4, type=int,     help='number of available GPU cards')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', help = 'alexnet or resnet18 or resnet20' )
parser.add_argument('--dataset', dest='dataset',     help='available datasets: 5datasets, miniimagenet, medmnist', default='5datasets', type=str)
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=64, type=int,  metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, metavar='M',     help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-world_size', '--world_size', default=4, type=int, help='total number of nodes')
parser.add_argument('-neighbors', '--neighbors', default=1, type=int, help='total number of neighbors of any node, added keeping in mind ring topology')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',   help='number of total epochs to run') 
parser.add_argument('--seed', default=1234, type=int,        help='set seed')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--print-freq', '-p', default=50, type=int,    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='save_temp', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='29500' , type=str)
parser.add_argument('--save-every', dest='save_every',  help='Saves checkpoints at every specified number of epochs',  type=int, default=5)
parser.add_argument('--biased', dest='biased', action='store_true',     help='biased compression')
parser.add_argument('--unbiased', dest='biased', action='store_false',     help='biased compression')
parser.add_argument('--level', default=32, type=int, metavar='k',  help='quantization level 1-32')
parser.add_argument('--eta',  default=1.0, type=float,  metavar='AR', help='averaging rate')  # default=1.0, and 0.0 means no sharing
parser.add_argument('--skew', default=0.0, type=float,     help='belongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--num_tasks', default=5, type=int,    help='number of tasks (over time)')
parser.add_argument('--graph', default='ring', type=str,    help='graph structure') 
parser.add_argument('--lamb', default=5000, type=int, help='lambda for EWC')
parser.add_argument('--clipgrad', default=100, type=int, help='gradient clipping')

args = parser.parse_args()

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i]=indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size    = int(skew*class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices
            
    
class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""
    def __init__(self, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        assert classes%tasks==0
        self.data = data
        self.partitions = {}
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False, num_workers=1) #change for miniimagenet
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
              labels = labels+targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices_full = sort_index.tolist()
        task_data_len = int(data_len/tasks)
        for n in range(tasks):
            ind_per_task = indices_full[n*task_data_len: (n+1)*task_data_len]
            indices_rand, indices = skew_sort(ind_per_task, skew=skew, classes=int(classes/tasks), class_size=class_size, seed=seed)
            self.partitions[n] = []
            for frac in sizes:
                if skew==1:
                    part_len = int(frac*task_data_len)
                    self.partitions[n].append(indices[0:part_len])  
                    indices = indices[part_len:]
                elif skew==0:
                    part_len = int(frac*task_data_len)
                    self.partitions[n].append(indices_rand[0:part_len]) 
                    if(args.eta!=0.0):
                        indices_rand = indices_rand[part_len:] #remove to use full data at each node for experiment
                else:
                    part_len = int(frac*task_data_len*skew); 
                    part_len_rand = int(frac*task_data_len*(1-skew))
                    part_ind = indices[0:part_len]+indices_rand[0:part_len_rand]
                    self.partitions[n].append(part_ind) 
                    indices = indices[part_len:]
                    indices_rand = indices_rand[part_len_rand:]

    def use(self, partition, task):
        return Partition(self.data, self.partitions[task][partition])

    
class DataPartition_5set(object):
    """ Partitions 5-datasets across different nodes, not setup for non-IID data yet, works only for SKEW=0"""
    def __init__(self, data_type, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        self.data = data
        self.partitions = {}
        indices_full = []
        data_len= []

        for i in range(len(data)):
            dataset = torch.utils.data.DataLoader(data[i], batch_size=512, shuffle=False, num_workers=2)
            data_len.append(len(data[i]))
            labels= []
            if(data_type=='5datasets'):
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    labels = labels+targets.tolist()
            else:
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    t = np.array(targets.tolist()).reshape(-1)
                    labels = labels+t.tolist()

            sort_index = np.argsort(np.array(labels))
            indices_full.append(sort_index.tolist())

        for n in range(tasks):
            task_data_len = int(data_len[n])
            ind_per_task = indices_full[n]
            rng = Random()
            rng.seed(seed)
            rng.shuffle(ind_per_task)
            self.partitions[n] = []
            for frac in sizes:
                part_len = int(frac*task_data_len)
                self.partitions[n].append(ind_per_task[0:part_len])
                if(args.eta!=0.0):
                    ind_per_task = ind_per_task[part_len:] #remove to use full data at each node for experiment

    def use(self, partition, task):
        return Partition(self.data[task], self.partitions[task][partition])

def partition_trainDataset(device,tasks=2):
    """Partitioning dataset"""

    if args.dataset == '5datasets':
        dataset= []
        classes= 10 #each task has 10 classes
        c= int(classes)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'Five_data/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,)
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'Five_data/',train=True,download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(), transforms.Normalize(mean,std)]))
        
        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'Five_data/SVHN',split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'Five_data/', train=True, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))

        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'Five_data/notmnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

    elif args.dataset == 'medmnist':
        #5-tasks: tissuemnist, organamnist, octmnist, pathmnist, bloodmnist
        classes= 11
        c= int(classes)

        # preprocessing
        data_transform = transforms.Compose([
            transforms.Pad(padding=2,fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        info = INFO['tissuemnist']
        DataClass = getattr(medmnist, info['python_class'])
        # load the data
        dataset_1 = DataClass(split='train', transform=data_transform, download=True)

        info = INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset_2 = DataClass(split='train', transform=data_transform, download=True)

        info = INFO['octmnist']
        DataClass = getattr(medmnist, info['python_class'])        
        dataset_3 = DataClass(split='train', transform=data_transform, download=True)

        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])  
        dataset_4 = DataClass(split='train', transform=data_transform, download=True)

        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])        
        dataset_5 = DataClass(split='train', transform=data_transform, download=True)

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

    elif args.dataset == 'miniimagenet':
        dataset= []
        classes= 100 #each task has 5 classes
        c= int(classes/tasks)
        class_size = {x:500 for x in range(100)} 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dataset= miniimagenet_setup.MiniImageNet(root='data_minii', train=True, transform=transforms.Compose([transforms.Resize((84,84)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

    size = dist.get_world_size()
    train_set={}
    if(args.eta==0.0):
        bsz = int((args.batch_size)) #exp for single agent setting in this setup (communication turned off)
        partition_sizes = [1.0 for _ in range(size)]
    else:
        bsz = int((args.batch_size) / float(size))
        partition_sizes = [1.0/size for _ in range(size)]

    if(dist.get_rank()==0):
        print("partition_sizes:", partition_sizes)

    if(args.dataset=='5datasets' or args.dataset=='medmnist'):
        partition= DataPartition_5set(args.dataset, dataset, partition_sizes, skew=args.skew, classes=classes, class_size=0, seed=args.seed, device=device, tasks=tasks)
    else:
        partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device, tasks=tasks)
    
    for n in range(tasks):
        task_partition = partition.use(dist.get_rank(), n)
        train_set[n]   = torch.utils.data.DataLoader(task_partition, batch_size=bsz, shuffle=True, num_workers=1)

    if(dist.get_rank()==0):
        print("len train set:", len(task_partition))

    return train_set, bsz, c


def test_Dataset_split(tasks):
    if args.dataset == '5datasets':

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'Five_data/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,)
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'Five_data/',train=False,download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False)
        for image, target in loader:
            image=image.expand(1,3,image.size(2),image.size(3))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'Five_data/SVHN',split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'Five_data/', train=False, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))

        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'Five_data/notmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

        val_set={}
        val_bsz = 64

        for n in range(tasks):
            task_data = dataset[n]
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5) #shuffle=False gives low test acc for bn with track_run_stats=False

    elif args.dataset == 'medmnist':

        # preprocessing
        data_transform = transforms.Compose([
            transforms.Pad(padding=2,fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        info = INFO['tissuemnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset_1 = DataClass(split='test', transform=data_transform, download=True)

        info = INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset_2 = DataClass(split='test', transform=data_transform, download=True)

        info = INFO['octmnist']
        DataClass = getattr(medmnist, info['python_class'])        
        dataset_3 = DataClass(split='test', transform=data_transform, download=True)

        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])  
        dataset_4 = DataClass(split='test', transform=data_transform, download=True)

        info = INFO['bloodmnist']
        DataClass = getattr(medmnist, info['python_class'])        
        dataset_5 = DataClass(split='test', transform=data_transform, download=True)

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

        val_set={}
        val_bsz = 64

        for n in range(tasks):
            task_data = dataset[n]
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5)

    elif args.dataset == 'miniimagenet':

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dataset= miniimagenet_setup.MiniImageNet(root='data_minii', train=False, transform=transforms.Compose([transforms.Resize((84,84)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels+targets.tolist()

        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len/tasks)

        val_bsz=10

        for n in range(tasks):
            ind_per_task = indices[n*task_data_len: (n+1)*task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=2)

    if(dist.get_rank()==0):
        print("len val_set:", len(task_data))

    return val_set, val_bsz

def fisher_matrix_diag(dataset, t, train_loader, model, model_old, criterion, device, c, sbatch=20):
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i,(input, target) in enumerate(train_loader):
        images, target = Variable(input).to(device), Variable(target%c).to(device)
        if(images.size(dim=1)==1):
            images= images.repeat(1, 3, 1, 1)
        if(dataset=='medmnist'):
            target = target.squeeze(1).to(dtype=torch.long)
        else:
            target = target.to(dtype=torch.long)
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion_ewc(t, outputs[t], target, criterion, model, model_old, fisher)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=images.size(dim=0)*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train_loader)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

def criterion_ewc(t, output, targets, criterion, model, model_old, fisher):
    # Regularization for all previous tasks
    loss_reg=0
    if t>0:
        for (name,param),(_,param_old) in zip(model.named_parameters(),model_old.named_parameters()):
            loss_reg+=torch.sum(fisher[name]*(param_old-param).pow(2))/2

    return criterion(output,targets)+args.lamb*loss_reg

def run(rank, size, q1, q2):
    global args, best_prec1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:{}".format(rank%args.devices))

    acc_matrix=np.zeros((args.num_tasks,args.num_tasks))
    prec_list = []
    best_prec1 = 0
    ##############
    data_transferred = []

    if(args.dataset == '5datasets'):
        task_details= [(0,10), (1,10), (2,10), (3,10), (4,10)]
    elif(args.dataset == 'medmnist'):
        task_details=[(0,8), (1,11), (2,4), (3,9), (4,8)]
    else:
        task_details = [(task,int(args.classes/args.num_tasks)) for task in range(args.num_tasks)]  # ex: [(0,5), (1,5)] for 2 tasks

    if(args.dataset == 'medmnist'):
        model= ResNet18(args.dataset, task_details, nf=32).to(device)
    else:
        model= ResNet18(args.dataset, task_details, nf=20).to(device)

    no_layers= 20
    
    if rank==0:
        print(args)
        print ('Model parameters ---')
        for k_t, (m, param) in enumerate(model.named_parameters()):
            print (k_t,m,param.shape)
        print ('-'*40) 
        print("*****EWC Fisher matrix calculated at all nodes, used after averaging******")
        if(args.dataset=='medmnist'):
            print("*********5-tasks: tissuemnist, organamnist, octmnist, pathmnist, bloodmnist**********")

    if(args.graph.lower()=='torus'):
        graph = GridGraph(rank, size, args.devices, peers_per_itr= args.neighbors) #Torus structure
    else:
        graph = RingGraph(rank, size, args.devices, peers_per_itr= args.neighbors) #undirected/directed ring structure

    if(rank==0):
        print(graph.get_peers())

    fisher = None
    model_old = None

    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank%args.devices],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device, 
                level = args.level,
                biased = args.biased,
                eta = args.eta,
                no_layers = no_layers,
                arch= args.arch,
                momentum=args.momentum,
                weight_decay = args.weight_decay,
                lr = args.lr,
                qgm = 0) 
    model.to(device)
    cudnn.benchmark = True
    train_loader, bsz_train, c = partition_trainDataset(device, args.num_tasks)
    val_loader, bsz_val        = test_Dataset_split(args.num_tasks)

    for task_id in range(0, args.num_tasks):
        data_per_task=0
        data_per_task_layer= np.zeros(no_layers)
        writer=0
        if(rank==0):
            print("************TASK*************:", task_id)
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum = args.momentum, nesterov=False)
        if rank==0 and task_id==0: print(optimizer)
        gamma= 0.1
        step1= int(args.epochs/2)
        step2= int(3/4*args.epochs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = gamma, milestones=[step1, step2])
        
        dist.barrier()
        if(task_id>0 and dist.get_rank()!=0):
            fisher_recv= q2.get()
            fisher= {}
            for f_key in fisher_recv.keys():
                fisher.update({f_key: torch.from_numpy(fisher_recv[f_key]).to(device)})

        dist.barrier()
        for epoch in range(0, args.epochs):  
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            model.block()

            dt, dt_layer, avg_loss= train(args.dataset, train_loader[task_id], model, model_old, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank, task_id, c, no_layers, fisher)
            data_per_task += dt
            data_per_task_layer= np.add(data_per_task_layer, dt_layer)

            lr_scheduler.step()
            prec1 = validate(args.dataset, val_loader[task_id], model, model_old, criterion, bsz_val, device, task_id, epoch, c, fisher)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

        data_per_task= data_per_task/1.0e9
        data_transferred.append(data_per_task)

        data_per_task_layer= data_per_task_layer/1.0e9

        if(rank==0):
            print("data transferred per task:", data_transferred)
            print("data transferred layerwise:", data_per_task_layer)
        
        if(args.eta!=0.0):
            dt= gossip_avg(args.dataset, train_loader[task_id], model, model_old, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank, task_id, c, fisher)
        else:
            print("no gossip averaging in case of turned off communication")

        x= validate(args.dataset, val_loader[task_id], model, model_old, criterion, bsz_val, device, task_id, epoch, c, fisher)
        model.eval()
        model_old=deepcopy(model.module)
        model_old.eval()
        freeze_model(model_old)

        if(dist.get_rank()==0 and task_id>0):
            fisher_old={}
            for n,_ in model.named_parameters():
                fisher_old[n]=fisher[n].clone()

        fisher= fisher_matrix_diag(args.dataset, task_id, train_loader[task_id], model, model_old, criterion, device, c) #fisher calculation at all nodes

        if(dist.get_rank()!=0):
            fisher_send= {}
            for f_key in fisher.keys():
                fisher_send.update({f_key: fisher[f_key].cpu().numpy()})     
            if(dist.get_rank()!=0):
                q1.put(fisher_send)
        dist.barrier()

        if(dist.get_rank()==0):
            fisher_avg= {}
            for n,p in model.named_parameters():
                fisher_avg[n]=0*p.data
            fisher_recv= []
            for i in range(args.world_size-1):
                fisher_recv.append(q1.get()) #other nodes fisher matrix
            for f_key in fisher.keys():
                for i in range(args.world_size-1):
                    fisher_avg[f_key]+=torch.from_numpy(fisher_recv[i][f_key]).to(device)
                fisher_avg[f_key]+=fisher[f_key]
                fisher_avg[f_key]=fisher_avg[f_key]/args.world_size #new updated average fisher matrix

            if task_id>0:
                for n,_ in model.named_parameters():
                    fisher_avg[n]=(fisher_avg[n]+fisher_old[n]*task_id)/(task_id+1)
            fisher= copy.deepcopy(fisher_avg) #keep updated fisher for node 0
            fisher_avg_send={}
            for f_key in fisher_avg.keys():
                fisher_avg_send.update({f_key: fisher_avg[f_key].cpu().numpy()})
            for i in range(args.world_size-1):
                q2.put(fisher_avg_send) #send the final updated fisher to all the other nodes!
        # test validation accuracy for all tasks
        jj = 0 
        prec= []
        for tn in range(task_id+1):
            acc_matrix[task_id,jj] = validate(args.dataset, val_loader[tn], model, model_old, criterion, bsz_val, device, tn, epoch, c, fisher) 
            prec.append(acc_matrix[task_id,jj])
            jj +=1
        prec_list.append(prec)
        print('Accuracies for node ', rank, '=')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()


    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))   

    total_data_transfer= 0
    if(rank==0):
        for i in range(len(data_transferred)):
            total_data_transfer= total_data_transfer+data_transferred[i]
        print("*****Total Data Transfer*****:", total_data_transfer)

def train(dataset, train_loader, model, model_old, criterion, optimizer, epoch, batch_size, lr, device, rank, task_id, c, no_layers, fisher):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0 
    data_layerwise = np.zeros(no_layers)
   
    # switch to train mode
    model.train()
    end = time.time()
    step = len(train_loader)*batch_size*epoch
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target%c).to(device)
        if(dataset=='medmnist'):
            target_var = target_var.squeeze(1).to(dtype=torch.long)
        else:
            target_var = target_var.to(dtype=torch.long)

        if(input_var.size(dim=1)==1):
            input_var= input_var.repeat(1, 3, 1, 1)   
        output = model(input_var)[task_id]
        loss= criterion_ewc(task_id, output, target_var, criterion, model, model_old, fisher)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clipgrad)
        optimizer.step()

        _, amt_data_transfer, amt_data_layerwise = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer

        for j in range(no_layers):
            data_layerwise[j]+=amt_data_layerwise[j]

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),  batch_time=batch_time,
                      loss=losses, top1=top1))
        step += batch_size 

    return data_transferred, data_layerwise, losses.avg

def gossip_avg(dataset, train_loader, model, model_old, criterion, optimizer, epoch, batch_size, lr, device, rank, task_id, c, fisher):
    """
       This function runs only gossip averaging for 50 iterations without local sgd updates - used to obtain the average model
    """
    data_transferred = 0 
    n = 50
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target%c).to(device)
        if(dataset=='medmnist'):
            target_var = target_var.squeeze(1).to(dtype=torch.long)
        else:
            target_var = target_var.to(dtype=torch.long)
        if(input_var.size(dim=1)==1):
            input_var= input_var.repeat(1, 3, 1, 1)
        # compute output
        output = model(input_var)[task_id]
        loss= criterion_ewc(task_id, output, target_var, criterion, model, model_old, fisher)
        loss.backward()
        optimizer.zero_grad()
        if(task_id==0):
            _, _, amt_data_transfer = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        else:
            _, _, amt_data_transfer = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        data_transferred += amt_data_transfer
        if i==n: break
    return data_transferred

def validate(dataset, val_loader, model, model_old, criterion, batch_size, device, task_id, epoch, c, fisher):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = len(val_loader)*batch_size*epoch

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader): 
            input_var, target_var = Variable(input).to(device), Variable(target%c).to(device)
            if(dataset=='medmnist'):
                target_var = target_var.squeeze(1).to(dtype=torch.long)
            else:
                target_var = target_var.to(dtype=torch.long)
            # compute output
            if(input_var.size(dim=1)==1):
                input_var= input_var.repeat(1, 3, 1, 1)
            output = model(input_var)[task_id] 
            loss= criterion_ewc(task_id, output, target_var, criterion, model, model_old, fisher)
            output = output.float()
            loss   = loss.float()
            # measure accuracy and record loss
            prec1  = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), 
                          loss=losses,
                          top1=top1))
            step += batch_size
    print(' * Prec@1 {top1.avg:.3f}' .format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_process(rank, size, fn, q1, q2, backend='nccl'):
    """Initialize distributed enviornment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size,q1,q2)

if __name__ == '__main__':
    size = args.world_size
    print(torch.cuda.device_count())
    manager= mp.Manager()
    q1= manager.Queue()
    q2= manager.Queue()
    spawn(init_process, args=(size,run,q1,q2), nprocs=size,join=True)
    

