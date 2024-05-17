import numpy as np

def Solve_TR(model, n, tr_radius, method_solve_tr, H = None, g = None):
    tr_sol = []
    if method_solve_tr == 'Dogleg' and H is not None and g is not None:
        # dogleg : to solve the TR subproblem
        pB = -np.dot(np.linalg.inv(H), g)  # the global minimal
        pU = -(g.T @ g) / (g.T @ H @ g) * g  # the gradient decent direction minimal

        if np.linalg.norm(pB) <= tr_radius:
            tr_sol = pB
        elif np.linalg.norm(pU) > tr_radius:
            tr_sol = -tr_radius / np.linalg.norm(g) * g
        else:
            pBU = pB - pU
            dotp = pU.T @ pBU
            tau_1 = ((dotp**2 - np.linalg.norm(pBU)**2 * (pU.T @ pU - tr_radius**2))**0.5 - dotp) / (pBU.T @ pBU)
            tr_sol = pU + tau_1 * pBU
        return tr_sol

    elif method_solve_tr == 'cauchy_point':
        # TODO: Implement cauchy point method
        pass
    elif method_solve_tr == 'tcg':
        # TODO: Implement truncated conjugate gradient method
        pass

    #tr_sol = np.ones(n)
    return tr_sol

Solve_TR([],[],1,'dogleg',H=np.diag(np.ones(3)),g=np.ones(3))