
"""
Mixing Manager Class

:description: Class provides an API for dynamically selecting mixing weights
              for gossip
"""


import torch


class MixingManager(object):

    def __init__(self, graph, device):
        self.graph_manager = graph
        self.device = device

    def is_regular(self):
        """
        Whether there is bias accumulated in local entry of stationary
        distribution of mixing matrix
        """
        return self.graph_manager.is_regular_graph() and self.is_uniform()

    def is_uniform(self):
        """ Whether mixing weights are distributed uniformly over peers """
        raise NotImplementedError

    def get_mixing_weights(self, residual_adjusted=True):
        """ Create mixing weight dictionary using uniform allocation """
        raise NotImplementedError


class UniformMixing(MixingManager):

    def get_mixing_weights(self):
        """ Create mixing weight dictionary using uniform allocation """
        mixing_weights = {}
        out_peers, _ = self.graph_manager.get_peers()
        mixing_weights['uniform'] = torch.tensor([1. / (len(out_peers)+1)], device=self.device)
        #mixing_weights['try'] = torch.tensor([1. / (len(out_peers)+2)], device=self.device)
        #print(mixing_weights)
        return mixing_weights

    def is_uniform(self): return True
