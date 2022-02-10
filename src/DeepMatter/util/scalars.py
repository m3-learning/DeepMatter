import numpy as np

class global_scaler:



    def fit(self, data):

    # calculate the mean and standard deviation of the input array
        self.mean = np.mean(data.reshape(-1))
        self.std = np.std(data.reshape(-1))

    def fit_transform(self, data):
        """

        :param data: the input array
        :type data: array
        :return: the data get through the normalization
        :rtype: array
        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        """

        :param data: the input data
        :type: array
        :return: the data get through the normalization
        :rtype: array
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """

        :param data: the normalized array
        :type: array
        :return: the same scale of the raw data
        :rtype: array
        """
        return (data * self.std) + self.mean

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
