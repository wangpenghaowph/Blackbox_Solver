import numpy as np

def Solve_TR(H, g, tr_radius, method_solve_tr):
    tr_sol = []
    """
    model is a quadratic function, solve for the hessian and gradient
    """
    
    if method_solve_tr == 'dogleg':
        # TODO: Implement dogleg method
        pass
    elif method_solve_tr == 'cauchy_point':
        # TODO: Implement cauchy point method
        pass
    elif method_solve_tr == 'tcg':
        # TODO: Implement truncated conjugate gradient method
        pass
    tr_sol = [1,1,1]
    return tr_sol
