def get_n_params(model):
    '''

    Function that gets the number of parameters in a pytorch model

    Args:
        model (object): pytorch model

    Returns: number of parameters in model

    '''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp