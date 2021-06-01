import numpy as np


class DimStandardScalar():
    '''
    Function that conducts standard scalar along a certain flatten dimension
    '''

    def __init__(self, axis=1):
        '''

        Args:
            axis: sets the axis to normalize the data
        '''
        self.axis = axis

    def fit(self, input):
        '''

        conducts the fit to normalize the data

        Args:
            input: data to fit

        Returns:

        '''
        self.mean_ = np.mean(input, axis=self.axis)
        self.mean_ = np.atleast_3d(self.mean_)
        if self.axis == 1:
            self.mean_ = np.transpose(self.mean_, (0, 2, 1))

        self.std_ = np.std(input, axis=self.axis)
        self.std_ = np.atleast_3d(self.std_)
        if self.axis == 1:
            self.std_ = np.transpose(self.std_, (0, 2, 1))

    def fit_transform(self, input):
        '''

        single function that fits and transforms the data

        Args:
            input: data to fit and transform

        Returns:

        '''
        self.fit(input)
        return self.transform(input)

    def transform(self, input):
        '''

        conduct the transform with a pre-developed model

        Args:
            input: data to fit and transform

        Returns:

        '''
        return (input - self.mean_) / self.std_
