import torch
import math


class Gaussian(object):

    def __init__(self, x_vector,
                 sd=[0, 1],
                 mean=[0, 1],
                 amp=[0, 1],
                 size=(1, 1),
                 verbose=False):
                 
        """ Class that computes the gaussian

        :param sd: range for the standard deviation
        :type sd: array, float
        :param mean: range for mean
        :type mean: array, float
        :param amp: range for the amplitude
        :type amp: array, float
        :param size: Size of the array first index is number of channels, second is number of functions.
        :type size: tuple
        :param verbose: shows outputs
        :type verbose: boolean
        
        """



        self.sd = sd
        self.mean = mean
        self.amp = amp
        self.x_vector = x_vector
        self.size = size
        self.verbose = verbose

    def compute(self, params, device='cpu'):
        """

        :param params:
        :type params:
        :param device: Sets the device to do the computation. Default `cpu`, common option `cuda`
        :type device: string, optional
        :return: spectra
        :rtype: Tensor
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

        :param device: evice where computation happens
        :type device: string
        :return: Generated spectra, parameters used for generation
        :rtype: Tensor, Tensor
        """

        sd = rand_tensor(min=self.sd[0], max=self.sd[1], size=self.size)
        mean = rand_tensor(min=self.mean[0], max=self.mean[1], size=self.size)
        amp = rand_tensor(min=self.amp[0], max=self.amp[1], size=self.size)

        _params = torch.torch.stack((mean, sd, amp))
        _params = torch.atleast_2d(_params)
        _params = torch.transpose((_params), 0, 1)
        return self.compute(_params, device=device), _params


def rand_tensor(min=0, max=1, size=(1)):
    """ Function that generates random tensor between a range of an arbitary size

    :param min:  sets the minimum value of parameter
    :type min: float
    :param max:  sets the maximum value of parameter
    :type max: float
    :param size: sets the size of the random vector to generate
    :type size: tuple
    :return: random tensor generated
    :rtype: tensor
    """

    out = (max - min) * torch.rand(size) + min
    return out
