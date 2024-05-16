import numpy as np

def Solve_TR(model, n, tr_radius, method_solve_tr):
    tr_sol = []
    if method_solve_tr == 'dogleg':
        # TODO: Implement dogleg method
        pass
    elif method_solve_tr == 'cauchy_point':
        # TODO: Implement cauchy point method
        pass
    elif method_solve_tr == 'tcg':
        # TODO: Implement truncated conjugate gradient method
        pass
    tr_sol = np.ones(n)
    return tr_sol
