import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .format import layout_fig, labelfigs
from matplotlib import pyplot as plt


def embedding_maps(data, image, colorbar_shown=True,
                   c_lim=None, mod=None,
                   title=None, fig_labels=None):
    """

    :param data: data need to be showed in image format
    :type data: array
    :param image: the output shape of the image
    :type image: array
    :param colorbar_shown: whether to show the color bar on the left of image
    :type colorbar_shown: boolean
    :param c_lim: Sets the scales of colorbar
    :type c_lim: list
    :param mod: set the number of image for each line
    :type mod: int
    :param title: set the title of figure
    :type title: string
    :param fig_labels: sets if figure labels are on
    :type fig_labels: bool
    :return: handel to figure being created
    :rtype: matplotlib figure
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels('')
            ax.set_yticklabels('')

            # adds the colorbar
        if colorbar_shown:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1f')

            # Sets the scales
            if c_lim is not None:
                im.set_clim(c_lim)

        if fig_labels is not None:
            labelfigs(ax, i, size=18)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16,
                     y=1, horizontalalignment='center')

    fig.tight_layout()

    return fig
