import os
import matplotlib.pyplot as plt

def make_folder(folder, **kwargs):
    """

    Args:
        folder: folder where to save
        **kwargs:

    Returns:
        folder: folder that was created

    """

    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return folder


def savefig(filename, eps=False, png=False, dpi=300):
    """
    function to save figures
    Args:
        filename: filename where to save figure
        eps: export as eps
        png: export as png
        dpi: selects the dots per inch

    Returns:

    """

    # Saves figures at EPS
    if eps:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=dpi, bbox_inches='tight')

    # Saves figures as PNG
    if png:
        plt.savefig(filename + '.png', format='png',
                    dpi=dpi, bbox_inches='tight')

