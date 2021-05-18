import torch

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
