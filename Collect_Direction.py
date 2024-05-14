import Sample
import numpy as np

def Collect_momentum(iter, history):
    #TODO:
    pass
def Collect_gradient(x, iter, history):
    pass
def Collect_DS(x, iter, history):
    pass    
def Collect_Direction(x, iter, mgs, history):
    n_m, n_g, n_f, n_all, n_s = mgs
    directions = {}
    # Collect the momentum direction
    if n_m:
        momentums = history.get_momentum_direction(n_m)
        directions['momentum'] = [momentums]
    if grad_fun:
        gradient = history.evaluate_grad(x, grad_fun, iter, 'EG')
        # Store the gradient as one of the directions
        directions['gradient'] = [gradient]
    else:
        points = Sample.Sample(x, 'uniform', 'DS', iter, history)
        # TODO: Estimate Gradient
    if n_all:
        # TODO:
        # Add other directions based on direct search or other heuristics
    return directions

