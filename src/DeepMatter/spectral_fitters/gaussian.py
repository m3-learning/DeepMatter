import torch
import math
from ..rand_util.rand_gen import rand_tensor

class Gaussian:
  """
  Class that computes the gaussian
  """

  def __init__(self, params, x_vector):
    """

    Args:
        self (object): Returns the instance itself.
        params (Tensor) : Array of values to compute
        x_vector (Tensor) : x-values to compute

    Returns:

    """

    self.params = params
    self.x_vector = x_vector

  def compute(self, device='cpu'):
    """

    Args:
        self (object): Returns the instance itself.
        device (string, optional) : Sets the device to do the computation. Default `cpu`, common option `cuda`

    Returns: out (Tensor): spectra.

    """
    _mean = self.params[:,0].to(device)
    _sd = self.params[:,1].to(device)
    _amp = self.params[:,2].to(device)
    x_vector = torch.Tensor.repeat(self.x_vector,self.params.shape[0]).reshape(self.params.shape[0],-1)
    x_vector = torch.swapaxes(x_vector,0,1).to(device)

    _temp = torch.tensor(math.pi * 2)
    _base = 1 / (_sd * (torch.sqrt(_temp)))
    _exp = torch.exp(-.5 * ((torch.pow((torch.sub(x_vector,_mean)),2)) / torch.pow(_sd,2)))
    return _amp*_base*_exp

  def sampler(self, length=1, device='cpu'):
    """

    Args:
      length: length of the vector to generate

    Returns:
      out (Tensor) : Generated spectra
      params (Tensor) : parameters used for generation

    """

    mean = self.params[:,0]
    sd = self.params[:,1]
    amp = self.params[:,2]

    sd = rand_tensor(min=sd[0], max=sd[1], size=length)
    mean = rand_tensor(min=mean[0], max=mean[1], size=length)
    amp = rand_tensor(min=amp[0], max=amp[1], size=length)

    _params = torch.torch.stack((mean, sd, amp))
    _params = torch.atleast_2d(_params)
    _params = torch.swapaxes((_params), 0, 1)
    return self.compute(device=device), _params