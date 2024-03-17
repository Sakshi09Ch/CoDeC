
"""
Collection of commonly used uitility functions
"""

import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist


def flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def unflatten(flat, tensor):
  
   offset=0
   numel = tensor.numel()
   output = (flat.narrow(0, offset, numel).view_as(tensor))
   return output


def group_by_dtype(tensors):
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def communicate(tensors, communication_op):
    """
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for dtype in tensors_by_dtype:
        flat_tensor = flatten_tensors(tensors_by_dtype[dtype])
        communication_op(tensor=flat_tensor)
        for f, t in zip(unflatten_tensors(flat_tensor, tensors_by_dtype[dtype]),
                        tensors_by_dtype[dtype]):
            t.set_(f)


def make_logger(rank, verbose=True):
    """
    Return a logger for writing to stdout;
    Arguments:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if not getattr(logger, 'handler_set', None):
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)  # prints to console
        logger.handler_set = True
    if not getattr(logger, 'level_set', None):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.level_set = True
    return logger


def is_power_of(N, k):
    """
    Returns True if N is a power of k
    """
    assert isinstance(N, int) and isinstance(k, int)
    assert k >= 0 and N > 0
    if k == 0 and N == 1:
        return True
    if k in (0, 1) and N != 1:
        return False

    return k ** int(round(math.log(N, k))) == N


def create_process_group(ranks):
    """
    Creates and lazy intializes a new process group. Assumes init_process_group
    has already been called.
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        new process group
    """
    initializer_tensor = torch.Tensor([1])
    if torch.cuda.is_available():
        initializer_tensor = initializer_tensor.cuda()
    new_group = dist.new_group(ranks)
    dist.all_reduce(initializer_tensor, group=new_group)
    return new_group
'''
def quantize_tensor(out_msg, comp_fn, quantization_level, is_biased = True):
    #print(quantization_level)
    out_msg_comp = copy.deepcopy(out_msg)
    quantized_values = comp_fn.compress(out_msg_comp, None, quantization_level, is_biased)
    #print(out_msg-quantized_values)
    
    return quantized_values

def quantize_layerwise(out_msg, comp_fn, quantization_level, is_biased = True):
    #print(quantization_level)
    quantized_values = []
    
    for param in out_msg:
        # quantize.
        #print(param.size())
        _quantized_values = comp_fn.compress(param, None, quantization_level, is_biased)
        quantized_values.append(_quantized_values)

    return quantized_values

def sparsify_layerwise(out_msg, comp_fn, comp_op, compression_ratio, is_biased=True):
    selected_values  = []
    selected_indices = []
    selected_shapes  = []
    
    for param in out_msg:
        p = flatten_tensors(param)
        values, indices = comp_fn.compress(p, comp_op, compression_ratio, is_biased)
        selected_values.append(values)
        selected_indices.append(indices)
        selected_shapes.append(len(values)) # should be same for all nodes, length of compressed tensor at each layer
        
    flat_values  = flatten_tensors(selected_values)
    flat_indices = flatten_tensors(selected_indices)
    comp_msg     = torch.cat([flat_values, flat_indices.type(flat_values.dtype)])
    return comp_msg, selected_shapes

def unsparsify_layerwise(msg, shapes, ref_param):
    # ref_msg is the out_msg from sparsify_layerwise.....need it just for the shape
    #sparse_msg = torch.zeros_like(ref_param)
    out_msg    = []
    val_size   = int(len(msg)/2)
    values     = msg[:val_size]
    indices    = msg[val_size:]
    indices    = indices.type(torch.cuda.LongTensor)
    
    pointer = 0
    i       = 0
    for ref in ref_param:
        param = torch.zeros_like(ref)
        p = flatten_tensors(param)
        layer_values  = values[pointer:(pointer+shapes[i])]
        layer_indices = indices[pointer:(pointer+shapes[i])]
        p[layer_indices] = layer_values.type(ref.data.dtype)
        layer_msg        = unflatten(p, ref)
        #print(layer_msg.size(), ref.size())
        out_msg.append(layer_msg)
        pointer  += shapes[i]
        i        += 1
        
    return out_msg
'''        
        
    
