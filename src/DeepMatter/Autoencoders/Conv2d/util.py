import numpy as np


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
