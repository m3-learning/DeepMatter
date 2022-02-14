import string
import numpy as np
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
def layout_fig(graph, mod=None, x=1, y=1):
    """
    function

    :param graph: number of axes to make
    :type graph: int
    :param mod: sets the number of figures per row
    :type mod: int (, optional)
    :return: fig:
                handel to figure being created
             axes:
                numpy array of axes that are created
    :rtype: fig:
                matplotlib figure
            axes:
                numpy array
    """

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
    if mod is None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod * x, y * 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)


def rotate_and_crop(image_, angle=60.46, frac_rm=0.17765042979942694):
    """
    function to rotate the image

    :param image_: image array to plot
    :type image_: array
    :param angle: angle to rotate the image by
    :type angle: float (, optional)
    :param frac_rm: sets the fraction of the image to remove
    :type frac_rm: float (, optional)
    :return: crop_image:
                 image which is rotated and cropped
             scale_factor:
                 scaling factor for the image following rotation
    :rtype: crop_image:
                 array
            scale_factor:
                 float
    """

    # makes a copy of the image
    image = np.copy(image_)
    # replaces all points with the minimum value
    image[~np.isfinite(image)] = np.nanmin(image)
    # rotates the image
    rot_topo = ndimage.interpolation.rotate(
        image, 90 - angle, cval=np.nanmin(image))
    # crops the image
    pix_rem = int(rot_topo.shape[0] * frac_rm)
    crop_image = rot_topo[pix_rem:rot_topo.shape[0] -
                                  pix_rem, pix_rem:rot_topo.shape[0] - pix_rem]
    # returns the scale factor for the new image size
    scale_factor = (np.cos(np.deg2rad(angle)) +
                    np.cos(np.deg2rad(90 - angle))) * (1 - frac_rm)

    return crop_image, scale_factor

def labelfigs(axes, number, style='wb', loc='br',
              string_add='', size=20, text_pos='center'):
    """
    Adds labels to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    number : int
        letter number
    style : str, optional
        sets the color of the letters
    loc : str, optional
        sets the location of the label
    string_add : str, optional
        custom string as the label
    size : int, optional
        sets the font size for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formatting_key = {'wb': dict(color='w',
                                 linewidth=1.5),
                      'b': dict(color='k',
                                linewidth=0.5),
                      'w': dict(color='w',
                                linewidth=0.5)}

    # Stores the selected option
    formatting = formatting_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    # Sets the location of the label on the figure
    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError(
            'Unknown string format imported please look at code for acceptable positions')

    # adds a custom string
    if string_add == '':

        # Turns to image number into a label
        if number < 26:

            axes.text(x_value, y_value, string.ascii_lowercase[number],
                      size=size, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axes.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=size, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:
        # writes the text to the figure
        axes.text(x_value, y_value, string_add,
                  size=size, weight='bold', ha=text_pos,
                  va='center', color=formatting['color'],
                  path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                       foreground="k")])

