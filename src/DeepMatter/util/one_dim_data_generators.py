import tensorflow as tf
import numpy as np

def non_linear_fn(t, x, y, z):
    # returns a function from variables
    return tf.nn.tanh(20 * (t - 2 * (x - .5))) + tf.nn.selu((t - 2 * (y - 0.5))) + tf.nn.sigmoid(-20 * (t - (z - 0.5)))


def generate_data(values, function=non_linear_fn, length=25, range_=[-1, 1]):
    """
    Function to generate data from values

    :param values: values to function for generating spectra
    :type values: float
    :param function:  mathematical expression used to generate spectra
    :type function: function, optional
    :param length: spectral length
    :type length: int (optional)
    :param range_: x range for function
    :type range_:  list of float
    :return: generatered spectra
    :rtype: array of float
    """

    # build x vector
    x = np.linspace(range_[0], range_[1], length)

    data = np.zeros((values.shape[0], length))

    for i in range(values.shape[0]):
        data[i, :] = function(x, values[i, 0], values[i, 1], values[i, 2])

    return data


def image_swatch_constructor(input, kernal_size=15, **kwargs):
    '''

    Args:
        input: image files to build swatches
        kernal_size: size of the kernal - only implemented as square
        **kwargs:  (x_range, y_range) sets the region to crop

    Returns:

    '''
    x_range = kwargs.get('x_range', [0, input.shape[1]])
    y_range = kwargs.get('y_range', [0, input.shape[2]])

    out = []

    for image in input:
        for i in range(x_range[0], x_range[1] - kernal_size):

            for j in range(y_range[0], y_range[1] - kernal_size):
                out.append(image[i:i + kernal_size, j:j + kernal_size])
    return np.array(out)
