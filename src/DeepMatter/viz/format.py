import matplotlib.pyplot as plt

def layout_fig(graph, mod=None):
    """

    Args:
        graph: Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
        mod: sets the number of figures per row

    Returns:
        fig : handle to figure being created.
        axes : numpy array of axes that are created.

    """

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
        elif graph < 64:
            mod = 8

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)