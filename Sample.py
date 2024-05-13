# This module is to sample a given number of points or a given number of directions.
"""
When do we use it?

In each iteration, there are three steps where we need sampling.
1. In stage "EG"(estimate gradient), we need to uniformly sample points.
2. In stage "DS"(direct search), we need to sample points either uniformly or orthogonally to the momentums and gradients.
3. In the stage "CM"(construct model), we need to sample points to construct the local model.
"""

def Sample_Direction(x, stage, num_points, ds='uniform'):
    results = []
    if method_search == 'orthogonal' and stage == 'DS':
        # TODO: 
        directions = []
        pass
    elif method_search == 'uniform':
        # TODO: 
        directions = []
    else:
        pass
    return directions

def Sample_Point(x, iter, stage, num_points, history, ds='uniform'):
    if stage == 'EG':
        #TODO:S
        points = []
    elif stage == 'DS':
        if ds == 'uniform':
            #TODO:
            points = [] 
        elif ds == 'orthogonal':
            #TODO:
            points = []
    elif stage == 'CM':
        # TODO:
        points = []
    return points