from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_n_params(model):
    '''

    Function that gets the number of parameters in a pytorch model

    Args:
        model (object): pytorch model

    Returns: number of parameters in model

    '''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class Dataset_Generator(Dataset):
    """
    Function that generates data
    """

    def __init__(self,
                 model,
                 samples_per_epoch=100000,
                 size=1,
                 **kwargs):
        self.model = model
        self.samples_per_epoch = samples_per_epoch
        self.sd = kwargs.get('sd')
        self.mean = kwargs.get('mean')
        self.amp = kwargs.get('amp')
        self.fraction = kwargs.get('fraction')
        self.x_vector = kwargs.get('x_vector')
        self.size = size
        self.noise = kwargs.get('noise')
        self.verbose = kwargs.get('verbose')
        self.function = self.model(self.x_vector,
                                   sd=self.sd,
                                   mean=self.mean,
                                   amp=self.amp,
                                   size=self.size)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        input, params = self.function.sampler(device='cuda')

        if self.noise is not None:
            if self.verbose is True and not None:
                print("adding noise\n")
            input += np.random.uniform(0, self.noise, size=input.shape)
            
        return {'input': input, 'params': params}

