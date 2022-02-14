import torch
import os
import tensorflow as tf
import numpy as np
import random

def rand_tensor(min=0, max=1, size=(1)):
    """
     Function that generates random tensor between a range of an arbitary size.

     Args:
         min (float): sets the minimum value of parameter
         max (float): sets the maximum value of the parameter
         size (tuple): sets the size of the random vector to generate

     Returns:
         out (pytorch.Tensor) : random tensor generated

     """
    out = (max - min) * torch.rand(size) + min
    return out

def set_seeds(seed=42):
    """
    :param seed: random value to set the sequence of the shuffle and random normalization

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
