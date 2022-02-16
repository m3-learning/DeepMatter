import string
import os
import numpy as np
from matplotlib import pyplot as plt
from os.path import join as pjoin


printing = {'PNG':True,
            'EPS':False,
           'dpi': 300}

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

def make_folder(folder, **kwargs):
    """
    Function that makes new folders

    :param folder: folder where to save
    :type folder: string
    :return: folder where to save
    :rtype: string
    """


    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return (folder)

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

def savefig(filename, printing):

    """
    function that saves the figure

    :param filename: path to save file
    :type filename: string
    :param printing: contains information for printing
                     'dpi': int
                            resolution of exported image
                      print_EPS : bool
                            selects if export the EPS
                      print_PNG : bool
                            selects if print the PNG
    :type printing: dictionary

    """


    # Saves figures at EPS
    if printing['EPS']:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=printing['dpi'], bbox_inches='tight')

    # Saves figures as PNG
    if printing['PNG']:
        plt.savefig(filename + '.png', format='png',
                    dpi=printing['dpi'], bbox_inches='tight')

