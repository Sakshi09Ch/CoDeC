

"""
Gossipers

:description: Gossiper's are designed for multi-peer communication (i.e., send
              and recv from multiple peers at each ieration)
"""

import torch
import torch.distributed as dist
import copy
from .mixing_manager import MixingManager
from .mixing_manager import UniformMixing
from .graph_manager import GraphManager
from .utils import (flatten_tensors)

import numpy as np

class dist_backend:
    UNDEFINED = -1
    TCP = 0
    MPI = 1
    GLOO = 2
    NCCL = 3


class Gossiper(object):
    """ Generic gossip averaging object for multi-peer communication """

    def __init__(self, msg, graph, device=None, logger=None,
                 rank=None, world_size=None, mixing=None,iters=1):
        """
        Initialize generic averaging class designed for multi-peer comms

        :param msg: (tensor) message used to initialize recv buffer
        :param device: (device) device on which to initialize recv buffer
        :param graph: (GraphManager) Subclass of GraphManager
        :param logger: (python logger) module used to log results
        """

        self.logger = logger
        self.iters = iters
        if rank is None or world_size is None:
            assert dist.is_initialized()
            # for now p2p communication only supported withed tcp and mpi
            assert dist._backend != dist_backend.GLOO
            assert dist._backend != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # graph topology properties
        self.rank = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager = graph
        self.peers_per_itr_device = torch.tensor(
            [self._graph_manager.peers_per_itr], device=device,
            dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(iteration=0, epoch=0, rotate=False)  # sets in- and out-peers attributes
        # mixing matrix
        if mixing is None:
            mixing = UniformMixing(self._graph_manager, device)
        assert isinstance(mixing, MixingManager)
        self._mixing_manager = mixing
        self.refresh_mixing_weights_()  # sets mixing-weights attribute
        
       
        # msg buffers used during send/recv
        self.device = device if device is not None else msg.device
        self.out_msg_buffer = []
        self.in_msg_buffer = msg.clone().detach_().to(self.device)

        if self.device.type == 'cpu':
            try:
                self.in_msg_buffer = self.in_msg_buffer.pin_memory()
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(e)
        self.placeholder = self.in_msg_buffer.clone()
        #print(self.placeholder.size())

        self._pending_req = None


    @property
    def peers_per_itr(self):
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self, iteration, epoch, rotate=None):
        """ Update in- and out-peers """
        # if rotate is None:
        #     rotate = True if self._graph_manager.is_dynamic_graph() else False
        # cannot cycle peers in a static graph
        #assert not (rotate and not self._graph_manager.is_dynamic_graph())
        if self._graph_manager.is_dynamic_graph():
            if iteration%self.iters==0:
                rotate = True
                #if self.rank==0: print('varying the graph at iteration', iteration)
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)
        
    def refresh_mixing_weights_(self, residual_adjusted=False):
        """ Update mixing-matrix weights """
        self.mixing_weights = self._mixing_manager.get_mixing_weights()
        
    def mix_out_msg_(self, out_msg, ps_weight):
        """ Returns a generator mixing messages on the fly """
        self.refresh_mixing_weights_()
        self.ps_weight = ps_weight
        out_msg = torch.cat([out_msg, self.ps_weight.type(out_msg.dtype)])
        # check whether or not we need to create a buffer for each out-msg
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform'] #1/2
            out_msg *= weight.type(out_msg.dtype)
            for _ in self.out_edges:
                yield out_msg
        
                
    def self_msg_(self, out_msg):
        """ Returns a generator messages on the fly """
        yield out_msg

    def clean_msg_buffers_(self):
        """ Clean outgoing message buffer """
        msgs = []
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msgs.append(msg)
        while len(msgs) > 0:
            msg = msgs.pop()
            with torch.no_grad():
                msg.set_()

    def parse_in_msg_buffer(self, residual=False):
        """ Parse in-msg buffer and return msg and ps-weight separately """
        msg = self.in_msg_buffer
        if not self.regular:
            return msg.narrow(0, 0, len(msg) - 1), msg[-1]
        else:
            if residual:
                return msg
            else:
                return msg

    def mix(self):
        """ Single gossip step """
        raise NotImplementedError



class SGD_DS(Gossiper):
    """ 1-peer Push-Sum consensus averaging module """

    def mix(self, out_msg, ref_msg, ps_weight, epoch, iteration, compress):
        """ Consensus averaging step """
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        # prepare messages for gossip
        placeholder = torch.zeros_like(torch.cat([out_msg, ps_weight]))

        if compress:
            self.placeholder= torch.zeros_like(placeholder)
            self.in_msg_buffer= torch.zeros_like(placeholder)

        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
       
        # non-blocking send
        data_amt = 0
        for out_edge in self.out_edges:
            msg = next(mixed_out_msgs)
            assert self.rank == out_edge.src
            req = dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))

            data_amt += msg.element_size()*msg.nelement()

        #receive 
        self.in_msg_buffer.zero_()
        for in_edge in self.in_edges:
            dist.broadcast(tensor=placeholder, src=in_edge.src, group=in_edge.process_group)
            
            # if(compress):
            #     print("receiving msg", self.placeholder.size())
            #     print()
            #     print("in_msg_buffer size:", self.in_msg_buffer.size())

            # if(compress and epoch in [0, 1e-3, 50, 50+1e-3]):
            #     print("received msg: ", self.placeholder.size())

            self.placeholder.copy_(placeholder) 
            self.in_msg_buffer.add_(self.placeholder)

                
        #print(epoch)
        self.refresh_peers_(epoch=epoch, iteration=iteration)
        self.clean_msg_buffers_()
        in_msg = self.in_msg_buffer.narrow(0, 0, len(self.in_msg_buffer) - 1)
        updated_ps_weight = self.in_msg_buffer[-1]

        return in_msg, updated_ps_weight, data_amt
