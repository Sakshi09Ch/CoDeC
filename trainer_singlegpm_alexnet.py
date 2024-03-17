import argparse
import os
import shutil
import time
import numpy as np
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
from random import Random

# Importing modules related to distributed processing
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn
#from torch.utils.tensorboard import SummaryWriter
###########
from gossip_choco import GossipDataParallel
from gossip_choco import RingGraph, GridGraph
from gossip_choco import UniformMixing
from gossip_choco import *
from models import *

parser = argparse.ArgumentParser(description='Propert AlexNet for CIFAR10/CIFAR100 in pytorch')
parser.add_argument('--devices', default=4, type=int,     help='number of available GPU cards')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alex_quarter', help = 'resnet or vgg or resquant' )
parser.add_argument('--dataset', dest='dataset',     help='available datasets: cifar10, cifar100', default='cifar100', type=str)
parser.add_argument('--classes', default=100, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=128, type=int,  metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, metavar='M',     help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-world_size', '--world_size', default=4, type=int, help='total number of nodes')
parser.add_argument('-neighbors', '--neighbors', default=1, type=int, help='total number of neighbors of any node, added keeping in mind ring topology')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',   help='number of total epochs to run') 
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
parser.add_argument('--compress',  default=False, type=bool,  metavar='COMP', help='True: compress by sending coefficients associated with the orthogonal basis space')  
parser.add_argument('--skew', default=0.0, type=float,     help='belongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--threshold', default=0.97, type=float,    help='threshold for the gradient memory')  # Similar to GPM-Codebase
parser.add_argument('--increment_th', default=0.003, type=float,    help='increase threshold linearly across tasks') 
parser.add_argument('--num_tasks', default=10, type=int,    help='number of tasks (over time)') #CIFAR-100 split into 10 tasks
parser.add_argument('--graph', default='ring', type=str,    help='graph structure') 

args = parser.parse_args()

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
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
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

    
def partition_trainDataset(device,tasks=2):
    """Partitioning dataset"""
    if args.dataset == 'cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        classes    = 10
        class_size = {x:5000 for x in range(10)}

        dataset = datasets.CIFAR10(root=f'data_cifar10', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        c = int(classes/tasks)
    elif args.dataset == 'cifar100':
        normalize  = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        classes    = 100
        class_size = {x:500 for x in range(100)}
        c = int(classes/tasks)

        dataset = datasets.CIFAR100(root=f'data_cifar100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

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

    partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device, tasks=tasks)
    
    for n in range(tasks):
        task_partition = partition.use(dist.get_rank(), n)
        train_set[n]   = torch.utils.data.DataLoader(task_partition, batch_size=bsz, shuffle=True, num_workers=5)

    return train_set, bsz, c


def test_Dataset_split(tasks):
    if args.dataset=='cifar10':
        
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])     
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=f'data_cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    val_set={}
    data_len = len(dataset)
    d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=5)
    labels = []
    for batch_idx, (inputs, targets) in enumerate(d):
        labels = labels+targets.tolist()
    sort_index = np.argsort(np.array(labels))
    indices = sort_index.tolist()
    task_data_len = int(data_len/tasks)
    val_bsz = 64
    
    for n in range(tasks):
        ind_per_task = indices[n*task_data_len: (n+1)*task_data_len]
        task_data = Partition(dataset, ind_per_task)
        val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5)

    return val_set, val_bsz


def run(rank, size, q1, q2):
    global args, best_prec1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:{}".format(rank%args.devices))
    task_details = [(task,int(args.classes/args.num_tasks)) for task in range(args.num_tasks)]  # ex: [(0,5), (1,5)] for 2 tasks
    acc_matrix=np.zeros((args.num_tasks,args.num_tasks))
    prec_list = []
    best_prec1 = 0
    ##############
    data_transferred = []
    if (args.arch=='alexnet'):
        model= alexnet(task_details).to(device)
    if(args.arch=='alex_quarter'):
        model=alexnet_scaled(task_details).to(device)
    no_layers= 5
    
    if rank==0:
        print(args)
        print ('Model parameters ---')
        for k_t, (m, param) in enumerate(model.named_parameters()):
            print (k_t,m,param.shape)
        print ('-'*40)  
        print("*****GPM calculation at node 0, broadcasted to all other nodes******")
    
    if(args.graph.lower()=='torus'):
        graph = GridGraph(rank, size, args.devices, peers_per_itr= args.neighbors) #Torus structure
    else:
        graph = RingGraph(rank, size, args.devices, peers_per_itr= args.neighbors) #undirected/directed ring structure based on neighbors

    if(rank==0):
        print(graph.get_peers())

    feature_list = []
    orth_basis= []

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
                compress = args.compress, 
                no_layers = no_layers,
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
        threshold = np.array([args.threshold] * 5) + task_id*np.array([args.increment_th] * 5)
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum = args.momentum, nesterov=False)
        if rank==0 and task_id==0: print(optimizer)
        gamma= 0.1
        step1= int(args.epochs/2)
        step2= int(3/4*args.epochs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = gamma, milestones=[step1, step2])
        
        feature_mat = []
        dist.barrier()

        if(dist.get_rank()!=0 and task_id>0):
            feature_list= q1.get()
            orth_basis= q2.get()
        
        if task_id>0:
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                if(rank==0):
                    print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            print ('-'*40)

        for epoch in range(0, args.epochs):  
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            model.block()

            dt, dt_layer, avg_loss= train(train_loader[task_id], model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank, feature_mat, task_id, c, no_layers, orth_basis, args.compress)
            data_per_task += dt
            data_per_task_layer= np.add(data_per_task_layer, dt_layer)

            lr_scheduler.step()
            prec1 = validate(val_loader[task_id], model, criterion, bsz_val, device, task_id, epoch, c)

        data_per_task= data_per_task/1.0e9
        data_transferred.append(data_per_task)

        data_per_task_layer= data_per_task_layer/1.0e9

        if(rank==0):
            print("data transferred per task:", data_transferred)
            print("data transferred layerwise:", data_per_task_layer)
        
        if(args.eta!=0.0):
            dt= gossip_avg(train_loader[task_id], model, criterion, optimizer, epoch, bsz_train, optimizer.param_groups[0]['lr'], device, rank, task_id, c, orth_basis, args.compress)
        else:
            print("no gossip averaging in case of turned off communication")
        
        # test validation accuracy for all tasks
        jj = 0 
        prec= []
        for tn in range(task_id+1):
            acc_matrix[task_id,jj] = validate(val_loader[tn], model, criterion, bsz_val, device, tn, epoch, c) 
            prec.append(acc_matrix[task_id,jj])
            jj +=1
        prec_list.append(prec)
        print('Accuracies for node ', rank, '=')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()

        if(dist.get_rank()==0):
            count, data_in = 0, None
            for i, (input, target) in enumerate(train_loader[task_id]):
                inp, target_in = Variable(input).to(device), Variable(target).to(device)
                data_in = torch.cat((data_in,inp),0) if data_in is not None else inp

                count += target_in.size(0)
                if count>=100: break
            mat_list = get_representation_matrix(model.module, device, data_in, 4, rank)

            if(args.eta==0.0):
                feature_list, orth_basis = update_GPM(mat_list, threshold, orth_basis, feature_list, args.compress, rank=rank, device=device)
            else:
                feature_list, orth_basis = update_GPM(mat_list, threshold, orth_basis, feature_list, args.compress, rank=rank, device=device)

            for nodes in range(args.world_size-1):
                q1.put(feature_list)
                q2.put(orth_basis)
        dist.barrier()
        
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))   

    total_data_transfer= 0
    if(rank==0):
        for i in range(len(data_transferred)):
            total_data_transfer= total_data_transfer+data_transferred[i]
        print("*****Total Data Transfer*****:", total_data_transfer)


def train(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank, feature_mat, task_id, c, no_layers, orth_basis, compress):
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
        # compute output
        output = model(input_var)[task_id]
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        loss.backward()
        
        if task_id>0:
            kk = 0 
            for k, (m,params) in enumerate(model.named_parameters()):
                if k<15 and len(params.size())!=1:

                    sz =  params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())
                    kk +=1
                elif (k<15 and len(params.size())==1) and task_id !=0 :
                    params.grad.data.fill_(0)
        
        optimizer.step()
        optimizer.zero_grad()

        if(task_id==0):
            _, amt_data_transfer,amt_data_layerwise = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr, orth_basis=orth_basis, compress=False)
        else:
            _, amt_data_transfer,amt_data_layerwise = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr, orth_basis=orth_basis, compress=compress)            
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

def gossip_avg(train_loader, model, criterion, optimizer, epoch, batch_size, lr, device, rank, task_id, c, orth_basis, compress):
    """
       This function runs only gossip averaging for 50 iterations without local sgd updates - used to obtain the average model
    """
    data_transferred = 0 
    n = 50
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target%c).to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output[task_id], target_var)
        loss.backward()
        optimizer.zero_grad()
        if(task_id==0):
            _, amt_data_transfer, _= model.transfer_params(epoch=epoch+(1e-3*i), lr=lr, orth_basis=orth_basis, compress=False)
        else:
            _, amt_data_transfer, _ = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr, orth_basis=orth_basis, compress=compress)
        data_transferred += amt_data_transfer
        if i==n: break
    return data_transferred

def validate(val_loader, model, criterion, batch_size, device, task_id, epoch, c):
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
            # compute output
            output = model(input_var)[task_id] 
            loss   = criterion(output, target_var)
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

def update_GPM (mat_list, threshold, orth_basis=[], feature_list=[], compress=False, rank=0, device=None):
    if(rank==0):
        print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])

            if compress:
                f_shape= np.shape(feature_list[i])
                M_MT =np.dot(feature_list[i],feature_list[i].transpose())
                I= np.identity(f_shape[0])
                Uo,So,Vo= np.linalg.svd(I-M_MT)
                orth_basis.append(Uo[:,0:f_shape[0]-f_shape[1]])

    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation 
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

            # criteria
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui

            if compress:
                f_shape= np.shape(feature_list[i])
                M_MT =np.dot(feature_list[i],feature_list[i].transpose())
                I= np.identity(f_shape[0])
                Uo,So,Vo= np.linalg.svd(I-M_MT)
                orth_basis[i]= Uo[:,0:f_shape[0]-f_shape[1]]

    if(rank==0):
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(feature_list)):
            print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
            if compress:
                print ('Orth Basis Layer {} : {}/{}'.format(i+1,orth_basis[i].shape[1], orth_basis[i].shape[0]))
        print('-'*40)
    return feature_list, orth_basis

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

def init_process(rank, size, fn, q1, q2,backend='nccl'):
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
    

