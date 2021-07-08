import torch
import math
from ..rand_util.rand_gen import rand_tensor

class PseudoVoigt:
    """
    Class that computes the voigt
    """

    def __init__(self, x_vector,
                 sd=[0, 1],
                 mean=[0, 1],
                 amp=[0, 1],
                 fraction=[0, 1],
                 size=(1, 1),
                 batch_size = 512,
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

        self.x_vector = x_vector
        self.batch_size=batch_size
        
        self.sd = sd
        self.sd_mean = torch.tensor(sd[0] + sd[1]) / 2
        self.sd_sd = torch.sqrt((torch.pow(torch.tensor(sd[1]) - torch.tensor(sd[0]), 2) / 12))
        
        self.mean = mean
        self.mean_mean = torch.tensor(mean[0] + mean[1]) / 2
        self.mean_sd = torch.sqrt((torch.pow(torch.tensor(mean[1]) - torch.tensor(mean[0]), 2) / 12))
        
        self.amp = amp
        self.amp_mean = torch.tensor(amp[0] + amp[1]) / 2
        self.amp_sd = torch.sqrt((torch.pow(torch.tensor(amp[1]) - torch.tensor(amp[0]), 2) / 12))
        
        self.fraction = fraction
        self.fraction_mean = torch.tensor(fraction[0] + fraction[1]) / 2
        self.fraction_sd = torch.sqrt((torch.pow(torch.tensor(fraction[1]) - torch.tensor(fraction[0]), 2) / 12))
        
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
            params = torch.reshape(params, (params.shape[0], 4, -1))

        if self.verbose:
            print(f'parm size = {params.size()}')
            print(f'x_vector size = {self.x_vector.shape[0]}')

        out = torch.zeros((params.shape[0],
                           self.x_vector.shape[0],
                           self.size[0],
                           self.size[1]))
        if self.verbose:
            print(self.size[1])
        params = params.to(device)
        for i in range(self.size[1]):
            _mean = params[:, 0, i] 
            _sd = params[:, 1, i]  + 1e-9
            _amp = params[:, 2, i] 
            _fraction = params[:, 3, i] 

            x_vector = torch.cat(params.shape[0] * [self.x_vector]).reshape(
                params.shape[0], -1).to(device)
            x_vector = torch.transpose(x_vector, 0, 1)#.to(device)
            
            if self.verbose:
                print(f'x_vector_shape = {x_vector.size()}')

            _log = torch.tensor(math.log(2))
            _sigma = _sd / torch.sqrt(2 * _log)
            _pi = torch.tensor(math.pi)
            _temp = ((1 - _fraction) * _amp) / (_sigma * torch.sqrt(_pi * 2))

            if self.verbose:
                print(f'mean_size = {_mean.size()}')

            _square = torch.pow((x_vector - _mean), 2)
            _exp = torch.exp( (0 - _square) / (2 * torch.pow(_sigma,2)))
            _left = _temp * _exp
            _temp = (_fraction * _amp) / _pi
            _right = _temp * (_sd / (_square + torch.pow(_sd, 2)))
            _out = _left + _right
            
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

        sd = rand_tensor(min=self.sd[0], max=self.sd[1], size=(self.batch_size, self.size[0],self.size[1]))
        mean = rand_tensor(min=self.mean[0], max=self.mean[1], size=(self.batch_size, self.size[0],self.size[1]))
        amp = rand_tensor(min=self.amp[0], max=self.amp[1], size=(self.batch_size, self.size[0],self.size[1]))
        fraction = rand_tensor(min=self.fraction[0], max=self.fraction[1], size=(self.batch_size, self.size[0],self.size[1]))
        _params = torch.torch.stack((mean, sd, amp, fraction))
        _params = torch.atleast_2d(_params)
        _params = torch.transpose((_params), 0, 1)
        _params = torch.transpose((_params), 1, 2)
        return _params,_params
        return self.compute(_params, device=device), _params
    