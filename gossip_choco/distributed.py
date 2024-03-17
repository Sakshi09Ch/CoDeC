
"""
Distributed Gossip Wrapper

:description: Multi-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer training.
"""

import functools
import time
import sys
import threading
import copy
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .gossiper import SGD_DS
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import UniformMixing
from .utils import (
    create_process_group, communicate, flatten_tensors,
    group_by_dtype, make_logger, unflatten_tensors, unflatten)

#from .compressor import QuantizationCompressor

#import global_init
HEARTBEAT_TIMEOUT = 1000  # maximum time to wait for message (seconds)


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, rank=None, world_size=None,
                 graph=None, mixing=None, comm_device=None, push_sum=True,
                 overlap=False, synch_freq=0, verbose=False, use_streams=False,
                 level=32, biased = False, eta = 0.5, momentum=0.9, lr=0.1, weight_decay = 1e-4, qgm=0, orth_basis=[], compress=False, no_layers=5, arch= 'alexnet'):
        super(GossipDataParallel, self).__init__()

        # devices available locally
        if device_ids is None:
            device_ids     = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids    = device_ids
        self.lr            = lr
        self.momentum      = momentum
        self.weight_decay  = weight_decay
    
        if world_size is None or rank is None:
            assert dist.is_initialized()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.process_rank = rank
        

        # put model on output device
        self.module = module
        first_param_dtype = next(self.module.parameters()).dtype
        
        # prepare local intra-node all-reduce objects
        self._module_copies = [self.module]

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = True if dist.get_backend() == 'gloo' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
        self.__cpu_comm = comm_device.type == 'cpu'
        

        if mixing is None:
            mixing = UniformMixing(graph, comm_device)
        self.mixing_weights = mixing.get_mixing_weights()
        self.averaging_rate   = torch.ones(1, device=comm_device).type(first_param_dtype)*eta

        self.average_iters    = int(1.0/eta) if eta > 0 else int(1.0)

        # distributed backend config
        self.dist_config = {
            'verbose':      verbose,
            'comm_device':  comm_device,
            'graph':        graph,
            'rank':         rank,
            'process_rank': self.process_rank,
            'world_size':   world_size,
            'cpu_comm':     self.__cpu_comm,
            'level':        level,
            'biased':       biased,
            'iters':        self.average_iters,
            'data_transferred': 0, 
            'data_layerwise': np.zeros(no_layers),
            'no_layers': no_layers,
            'epoch':            0.0,
            'iterations':         0,
            'gossip': 0,
            'compress': False,
            'orth_basis':[],
            'arch': arch}
            
        self.overlap = overlap
        self.synch_freq = synch_freq
        self.num_updates = 0
        self.asynch = synch_freq > 0

        # logger used to print to stdout
        self.logger = make_logger(rank, verbose)  

        # prepare parameters for gossip
        self.gossip_enable = True
        self.gossiping = False
        self.params_mixed = True
        self.is_ps_numerator = False
        
        
        self.gossip_ps_weight = torch.ones(1, device=comm_device).type(first_param_dtype)
        self.ps_weight        = torch.ones(1, device=comm_device).type(first_param_dtype)
        self.quasi_global_momentum = qgm
        self.gossip_params        = []
        self.gossip_device_buffer = []
        self.neighbor_copy        = []
        self.self_xhat            = []
        #buffers for qgm 
        self.momentum_buff        = []
        
        for p in module.parameters():
            cp = p.clone().detach_()
            cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.to(comm_device)
            self.gossip_params.append(cp)
            self.gossip_device_buffer.append(cp)
            self.neighbor_copy.append(torch.zeros_like(cp).to(comm_device))
            self.self_xhat.append(torch.zeros_like(cp).to(comm_device))
            self.momentum_buff.append(torch.zeros_like(cp).to(comm_device))
            
        # prepare gossip process control objects
        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()

        if self.dist_config['comm_device'].type != 'cpu' and use_streams:
            self.gossip_stream = torch.cuda.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream(device=comm_device)

        
        self.gossip_thread = threading.Thread(
                target=GossipDataParallel._gossip_target,
                args=(self.dist_config,
                      self.gossip_flag,
                      self.train_flag,
                      self.gossip_lock,
                      self.gossip_params, #SEND
                      self.gossip_device_buffer,
                      self.gossip_stream,
                      self.self_xhat,
                      self.gossip_ps_weight))
        self.gossip_thread.daemon = True
        self.gossip_thread.name = 'Gossip-Thread'
        self.gossip_thread.start()
      
        # wait for thread to complete initialization
        self.gossip_flag.wait()
        self.gossip_flag.clear()

        # register ps/grad-reduction hooks
        self.__register_hooks()

    def update_gossiper(self, attr, val):
        self.logger.debug('waiting for gossip lock')
        with self.gossip_lock:
            self.logger.debug('gossip lock received')
            for gossiper in self.dist_config['gossipers'].values():
                if val == getattr(gossiper, attr):
                    self.logger.debug('nothing to update')
                    return
                # update attr
                self.logger.debug('setting gossiper {} to {}'.format(attr, val))
                setattr(gossiper, attr, val)

    def state_dict(self, finish_gossip=True):
        if finish_gossip:
            self._query_gossip_queue()

        super_dict = super(GossipDataParallel, self).state_dict()
        supplanted_dict = {'state_dict': super_dict,  }
        return supplanted_dict

    def load_state_dict(self, load_dict):
        state_dict = load_dict['state_dict']
        super(GossipDataParallel, self).load_state_dict(state_dict)
       
    def ps_numerator(self):
        """ Convert model params to ps-numerator """
        if not self.is_ps_numerator:
            ps_weight = self.ps_weight 
            for p in self.module.parameters():
                p.data.mul_(ps_weight.type(p.data.dtype))
            
            if self.quasi_global_momentum==1:
                for p, m in zip(self.module.parameters(), self.momentum_buff):
                    if p.requires_grad:
                      buff = copy.deepcopy(m.data)
                      p.grad.data.add_(p.data, alpha = self.weight_decay)
                      buff.data.mul_(self.momentum).add_(p.grad.data)
                      p.grad.data.add_(buff.data, alpha = self.momentum)
                      m.data.mul_(self.momentum).add_(p.grad.data, alpha=1-self.momentum)     
            self.is_ps_numerator = True
                
    def unbias(self):
        """ Convert model params to de-biased estimate """
        if self.is_ps_numerator:
            ps_weight = self.ps_weight
            for p in self.module.parameters():
                p.data.div_(ps_weight.type(p.data.dtype))
            self.is_ps_numerator = False
            
    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module(*inputs[0], **kwargs[0])
        
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)


    def train(self, mode=True):
        super(GossipDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(GossipDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()
        self._query_gossip_queue(non_blocking=self.asynch)

    def block(self):
        self.logger.info('blocking')
        dist.barrier()
    
    def update_hats(self):
        for xj_hat, xi_hat, qj in zip(self.neighbor_copy, self.self_xhat, self.gossip_device_buffer): # change to have weighted q
            xj_hat.data.add_(qj.data)
            qj.data.copy_(xj_hat.data)
            qj.data.add_(xi_hat.data, alpha = float(self.mixing_weights['uniform'])-1.0)
        
    def _query_gossip_queue(self, non_blocking=False):
        """ Check gossip-queue for push-sum residuals and update model """
        if not self.gossip_enable:
            return

        self.logger.debug('querying gossip queue')

        # no gossip happening right now so just return
        if not self.gossiping:
            #self.logger.warning('not gossiping right now')
            return False

        if not non_blocking:
            if not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
                raise NameError('Gossip flag timeout')
                sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug('received gossip flag')
            #goss = []
               
            self.gossip_ps_weight.data.add_(self.ps_weight.data, alpha = float(self.mixing_weights['uniform'])-1.0)
            self.gossip_ps_weight.data.mul_(self.averaging_rate.type(self.ps_weight.data.dtype))
            self.ps_weight.data.add_(self.gossip_ps_weight)
            self.update_hats()
              
            #rec_i=0
            for p, r, m in zip(self.module.parameters(), self.gossip_device_buffer, self.momentum_buff):
                r.data.mul_(self.averaging_rate.type(r.data.dtype))

                #g = copy.deepcopy(r.data).view(-1) 
                #goss.extend(g)
                p.data.add_(r.data) 
                if self.quasi_global_momentum==1 and p.requires_grad:
                    m.data.add_(r.data, alpha=((self.momentum-1.0)/(self.lr*float(self.ps_weight.data))))     
                
            # update flags
            #self.logger.debug('updated ps-weight {}'.format(self.ps_weight))
            self.logger.debug('updated model params')
            self.gossip_ps_weight.copy_(self.ps_weight)
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True

    def transfer_params(self, mix=True, epoch=0, lr=0.1, orth_basis=[], compress=False):
        """ Transfers COPY of model parameters to gossip queue """
        self.lr=lr
        self.dist_config['orth_basis']= orth_basis
        self.dist_config['compress']= compress
        self.dist_config['iterations']+=1
        if (not self.gossip_enable ):
            return False, 0

        self.logger.debug('transfering model params')

        # don't transfer new params if old params haven't been mixed yet
        if not self.params_mixed:
            self.logger.warning('params not mixed')
            return False, 0
        # params gpu-gpu copy (fast)
        # compute xi-xi_hat
        for xi, xi_hat, gossip_device_buffer_elem in zip(self.module.parameters(), self.self_xhat, self.gossip_device_buffer):
            gossip_device_buffer_elem.data.copy_(xi.data)
            gossip_device_buffer_elem.data.add_(xi_hat.data, alpha=-1)
             
        # --
        # buffer to gossip-thread copy (potentially slow, but asynchronous)
        # --
        self.gossip_stream.wait_stream(torch.cuda.current_stream(device = self.dist_config['comm_device']))
        
        with torch.cuda.stream(self.gossip_stream):
            for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
                gp.copy_(b, non_blocking=True)
                
        # update flags
        self.logger.debug('transfered model params')
        self.dist_config['epoch'] = epoch
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True, self.dist_config['data_transferred'], self.dist_config['data_layerwise']

    @staticmethod
    def _gossip_into_receive_buffer(send_buffer, 
                                    gossiper, 
                                    receive_buffer,
                                    gossip_lock,
                                    dist_config, self_xhat, gossip_ps_weight):

        orth_b = dist_config['orth_basis']
        send_total= []
        sent_layerwise= np.zeros(dist_config['no_layers']) #data sent per GPM layer
        if(dist_config['arch']=='alexnet' or dist_config['arch']=='alex_quarter'):
            gpm_l= [0, 3, 6, 9 ,12] #specific to AlexNet, layers with GPM
        elif(dist_config['arch']=='resnet18'):
            gpm_l= [0, 3, 6, 9 ,12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57] #specific to ResNet18, layers with GPM

        if(dist_config['compress']):
            for i in range(len(send_buffer)):
                if(i in gpm_l):
                    orth_T= torch.t(torch.Tensor(orth_b[int(i/3)]).to(dist_config['comm_device']))
                    send_buff_copy= send_buffer[i].clone()
                    send_buff_reshape= send_buff_copy.view(send_buff_copy.size(0), -1) #Cout*(Ci*k*k)
                    send_en= torch.matmul(orth_T, torch.t(send_buff_reshape))
                    send_total.append(send_en) #encoded
                    sent_layerwise[int(i/3)]= send_en.element_size()*send_en.nelement()
                else:
                    send_total.append(send_buffer[i]) #original because no GPM for these layers
      
            comp_msg=flatten_tensors(send_total) #coefficients and original updates sent layerwise

        else:
            comp_msg = flatten_tensors(send_buffer) #message to be sent by default
            for i in range(len(send_buffer)):
                if(i in gpm_l):
                    sent_layerwise[int(i/3)]= send_buffer[i].element_size()*send_buffer[i].nelement()

        # update xi_hat            
        for xi_hat, qi in zip(self_xhat, send_buffer):
            xi_hat.data.add_(qi.data)
          
        with gossip_lock:
            data_amt = 0
            #in_msg = sum of q_j's
            in_msg, updated_ps_weight, data_amt = gossiper.mix(comp_msg, send_buffer, gossip_ps_weight, epoch=dist_config['epoch'], iteration=dist_config['iterations'], compress= dist_config['compress'])
            dist_config['data_transferred'] = data_amt
            dist_config['data_layerwise'] = sent_layerwise

        if(dist_config['compress']):
            i=0
            for r, g in zip(unflatten_tensors(in_msg, send_total), receive_buffer):
                if (i in gpm_l):
                    r_decoded= torch.matmul(torch.Tensor(orth_b[int(i/3)]).to(dist_config['comm_device']), r)
                    r_decoded= torch.t(r_decoded)
                    r_decoded= torch.reshape(r_decoded, g.size())
                else:
                    r_decoded= r

                if dist_config['cpu_comm']:
                    g.copy_(r_decoded, non_blocking=True)
                else:
                    g.data.copy_(r_decoded)
                i+=1
        else: 
            for r, g in zip(unflatten_tensors(in_msg, send_buffer), receive_buffer):
                if dist_config['cpu_comm']:
                    g.copy_(r, non_blocking=True)
                else:
                    g.data.copy_(r)
            
        return updated_ps_weight

    @staticmethod
    def _gossip_target(dist_config, gossip_flag, train_flag, gossip_lock,
                       gossip_params, gossip_device_buffer,
                       gossip_stream, self_xhat, gossip_ps_weight):
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(dist_config['rank'], dist_config['verbose'])

        gossip_params_by_dtype        = group_by_dtype(gossip_params) #send
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer) #receive
        self_xhat_by_dtype            = group_by_dtype(self_xhat) #xi_hat

        gossipers = {}
        # init gossip instance
        gossiper_class = SGD_DS
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(
                torch.cat([flatten_tensors(gossip_params_by_dtype[dtype]), gossip_ps_weight]), 
                device=dist_config['comm_device'],
                graph=dist_config['graph'],
                rank=dist_config['process_rank'],
                world_size=dist_config['world_size'], iters=dist_config['iters'],
                logger=logger)

        dist_config['gossipers'] = gossipers
        
        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                if True:
                #with torch.cuda.stream(gossip_stream):
                    for dtype in gossip_params_by_dtype:
                        #assert not torch.isnan(flatten_tensors(gossip_params_by_dtype[dtype])).any()
                        ps_weight = GossipDataParallel._gossip_into_receive_buffer(
                            gossip_params_by_dtype[dtype], gossipers[dtype],
                            gossip_device_buffer_by_dtype[dtype], 
                            gossip_lock, dist_config, self_xhat_by_dtype[dtype], gossip_ps_weight)
                        gossip_ps_weight.copy_(ps_weight)
                        
                    
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
                for gossiper in gossipers.values():
                    gossiper.clean_msg_buffers_()
                
            finally:
                gossip_stream.synchronize()
                # give main thread go-ahead to read our gossip buffer
                train_flag.clear()
                gossip_flag.set()

    def __register_hooks(self):
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self):
        self.logger.debug('making backward hook')

        def hook(*unused):
            # reduce gradients across devices on a single machine
            if len(self.device_ids) > 1:

                # collect gradients from all copies
                all_grads = [[] for _ in range(len(self._module_copies))]
                for dev_idx, module in enumerate(self._module_copies):
                    for p in module.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        all_grads[dev_idx].append(p.grad.data)

                # reduce grads
                reduced_grads = reduce_add_coalesced(
                    all_grads, self.output_device,
                    self.nccl_reduce_bucket_size)

                # update grads with reduced grads
                for grad, reduced in zip(all_grads[0], reduced_grads):
                    grad.copy_(reduced)

                # clear the gradients and parameters across all replicas
                for module in self._module_copies[1:]:
                    for param in module.parameters():
                        if param.requires_grad:
                            param.grad = None
                            with torch.no_grad():
                                param.set_()
            # convert model back to ps-numerator
            self.ps_numerator()
                                
        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def __make_forward_pre_hook(self):
        self.logger.debug('making forward pre-hook')

        def hook(*unused):
            """ Query gossip queue and de-bias during forward pass """
            # gossip during training (not inference)
            if self.gossip_enable:
                non_blocking = self.num_updates < self.synch_freq
                if self._query_gossip_queue(non_blocking):
                    self.num_updates = 0
                else:
                    self.num_updates += 1
                if self.overlap:
                    self.transfer_params()
            #convert model to de-biased estimate
            self.unbias()

        return hook
