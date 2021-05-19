import torch
import math
from ..rand_util.rand_gen import rand_tensor


class Gaussian:
    """
  Class that computes the gaussian
  """

    def __init__(self, x_vector,
                 sd=[0, 1],
                 mean=[0, 1],
                 amp=[0, 1]):
        """

    Args:
      sd (array, float): range for the standard deviation
      mean (array, float): range for the mean
      amp (array, float): range for the amplitude
      x_vector (array, float): array to plot

    """

        self.sd = sd
        self.mean = mean
        self.amp = amp
        self.x_vector = x_vector

    def compute(self, params, device='cpu'):
        """

    Args:
        self (object): Returns the instance itself.
        device (string, optional) : Sets the device to do the computation. Default `cpu`, common option `cuda`

    Returns: out (Tensor): spectra.

    """
        _mean = params[:, 0].to(device)
        _sd = params[:, 1].to(device)
        _amp = params[:, 2].to(device)
        x_vector = torch.Tensor.repeat(self.x_vector, params.shape[0]).reshape(params.shape[0], -1)
        x_vector = torch.transpose(x_vector, 0, 1).to(device)

        _temp = torch.tensor(math.pi * 2)
        _base = 1 / (_sd * (torch.sqrt(_temp)))
        _exp = torch.exp(-.5 * ((torch.pow((torch.sub(x_vector, _mean)), 2)) / torch.pow(_sd, 2)))
        return _amp * _base * _exp

    def sampler(self, length=1, device='cpu'):
        """

    Args:
      length: length of the vector to generate

    Returns:
      out (Tensor) : Generated spectra
      params (Tensor) : parameters used for generation

    """

        sd = rand_tensor(min=self.sd[0], max=self.sd[1], size=length)
        mean = rand_tensor(min=self.mean[0], max=self.mean[1], size=length)
        amp = rand_tensor(min=self.amp[0], max=self.amp[1], size=length)

        _params = torch.torch.stack((mean, sd, amp))
        _params = torch.atleast_2d(_params)
        _params = torch.transpose((_params), 0, 1)
        return self.compute(_params, device=device), _params
