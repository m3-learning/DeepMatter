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
                 amp=[0, 1],
                 size=(1, 1),
                 verbose=False):
        """

        Args:
            x_vector:
            sd (array, float): range for the standard deviation
            mean (array, float): range for the mean
            amp (array, float): range for the amplitude
            size (tuple): Size of the array first index is number of channels, second is number of functions
            verbose (bool): shows outputs
        """


        self.sd = sd
        self.mean = mean
        self.amp = amp
        self.x_vector = x_vector
        self.size = size
        self.verbose = verbose

    def compute(self, params, device='cpu'):
        """

        Args:
            self (object): Returns the instance itself.
            device (string, optional) : Sets the device to do the computation. Default `cpu`, common option `cuda`

        Returns: out (Tensor): spectra.

        """

        if self.verbose:
            print({f'pre-param size {params.size()}'})

        if len(params.size()) == 2:
            params = torch.reshape(params, (params.shape[0], 3, -1))

        if self.verbose:
            print(f'parm size = {params.size()}')
            print(f'x_vector size = {self.x_vector.shape[0]}')

        out = torch.zeros((params.shape[0],
                           self.x_vector.shape[0],
                           self.size[0],
                           self.size[1]))
        if self.verbose:
            print(self.size[1])

        for i in range(self.size[1]):
            _mean = params[:, 0, i].to(device)
            _sd = params[:, 1, i].to(device)
            _amp = params[:, 2, i].to(device)

            x_vector = torch.cat(params.shape[0] * [self.x_vector]).reshape(
                params.shape[0], -1).to(device)

            x_vector = torch.transpose(x_vector, 0, 1).to(device)

            if self.verbose:
                print(f'x_vector_shape = {x_vector.size()}')

            _temp = torch.tensor(math.pi * 2)
            _base = 1 / (_sd * (torch.sqrt(_temp)))

            if self.verbose:
                print(f'mean_size = {_mean.size()}')
            _exp = torch.exp(-.5 * ((torch.pow((torch.sub(x_vector, _mean)), 2)) / torch.pow(_sd, 2)))
            _out = _amp * _base * _exp
            if self.verbose:
                print(f'amp {_amp.size()}, base {_base.size()}, exp {_exp.size()}')
                print(f'out shape = {_out.shape}')
            out[:, :, 0, i] = torch.transpose(_out, 0, 1)

        return torch.sum(out, dim=3)

    def sampler(self, device='cpu'):
        """

        Args:
            device (str): device where computation happens

        Returns:
            out (Tensor) : Generated spectra
            params (Tensor) : parameters used for generation

        """

        sd = rand_tensor(min=self.sd[0], max=self.sd[1], size=self.size)
        mean = rand_tensor(min=self.mean[0], max=self.mean[1], size=self.size)
        amp = rand_tensor(min=self.amp[0], max=self.amp[1], size=self.size)

        _params = torch.torch.stack((mean, sd, amp))
        _params = torch.atleast_2d(_params)
        _params = torch.transpose((_params), 0, 1)
        return self.compute(_params, device=device), _params
