import numpy as np


def find_switching(embeddings, size=(129, 64, 190, 64), **kwargs):
    """
    Function that identifies the points where switching occurs

    Args:
        embeddings: the computed embeddings
        size: the size of the array
        **kwargs:

    Returns:
        movie: the movie

    """
    embeddings_ = embeddings.reshape(size)

    kwargs.setdefault('ranges_', embeddings_.shape[2])
    ranges_ = kwargs.get('ranges_')

    def find_largest_time_step(images, start, end, i, j):
        times = start
        record = times
        error = 0
        while times < end:
            test_error = np.sum(abs(images[times + 1, i, j] - images[times, i, j]))
            if test_error > error:
                record = times + 1
                error = test_error
            times += 1

        return record

    movie = np.zeros(size[0:3])

    switch_ind = []

    # loops around y position
    for i in range(size[1]):

        # loops around x position
        for j in range(size[2]):

            # loops around all of the ranges selected
            for ranges__ in ranges_:
                switch_ind.append(find_largest_time_step(embeddings_,
                                                         ranges__[0], ranges__[1],
                                                         i, j))

    switch_ind = np.array(switch_ind).reshape(size[1], size[2], len(ranges_))

    # loops around y position
    for i in range(size[1]):

        # loops around x position
        for j in range(size[2]):

            ranges__ = switch_ind[i, j]

            # selects the even values and makes one
            for k in range(len(ranges__) - 1):
                if (k % 2) == 0:
                    movie[ranges__[k]:ranges__[k + 1], i, j] = 1

                # for the odd values makes 0
                else:
                    movie[ranges__[k]:ranges__[k + 1], i, j] = 0

    return movie, switch_ind
