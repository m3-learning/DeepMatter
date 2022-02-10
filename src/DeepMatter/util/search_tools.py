import numpy as np

def find_nearest(array, value, averaging_number):
    """
    function to find the index of the nearest value in the array

    :param array: image to find the index closest to a value
    :type array: float, array
    :param value: value to find points near
    :type value: float
    :param averaging_number: number of points to find
    :type averaging_number: int
    :return: returns the indices nearest to a value in an image
    :rtype: array
    """

    idx = (np.abs(array-value)).argsort()[0:averaging_number]
    return idx