import os
import matplotlib.pyplot as plt

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

