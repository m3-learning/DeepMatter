import os


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
