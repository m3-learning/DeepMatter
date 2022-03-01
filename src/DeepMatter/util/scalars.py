import numpy as np


class global_scaler:

    def fit(self, data):
        """

        :param data: data go to the fit function
        :type data: numpy array
        """

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
        :type: numpy array
        :return: the data get through the normalization
        :rtype: numpy array
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """

        :param data: the normalized array
        :type: numpy array
        :return: the same scale of the raw data
        :rtype: numpy array
        """
        return (data * self.std) + self.mean


class DimStandardScalar():
    '''
    Function that conducts standard scalar along a certain flatten dimension
    '''

    def __init__(self, axis=1):
        """

        :param axis: set the axis to mormalize the data
        :type axis: int
        """
        self.axis = axis

    def fit(self, input):
        """

        :param input: data to fit
        :type input: numpy array

        """
        self.mean_ = np.mean(input, axis=self.axis)
        self.mean_ = np.atleast_3d(self.mean_)
        if self.axis == 1:
            self.mean_ = np.transpose(self.mean_, (0, 2, 1))

        self.std_ = np.std(input, axis=self.axis)
        self.std_ = np.atleast_3d(self.std_)
        if self.axis == 1:
            self.std_ = np.transpose(self.std_, (0, 2, 1))

    def fit_transform(self, input):
        """

        :param input: data to fit and transform
        :type input: numpy array
        :return: fitted and transformed data
        :rtype: numpy array
        """
        self.fit(input)
        return self.transform(input)

    def transform(self, input):
        """

        :param input: data for transform
        :type input: numpy array
        :return: data transformed
        :rtype: numpy array
        """
        return (input - self.mean_) / self.std_
