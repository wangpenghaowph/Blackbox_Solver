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
    
    if method_solve_tr == 'Dogleg' and H is not None and g is not None:
        epsilon = 1e-4
        max_iter = self.sub_dim

        s = np.zeros((self.sub_dim,1))
        r = self.g
        r_norm_0 = np.linalg.norm(r)
        p = -self.g
        k = 0
        while k<max_iter:
            if p.T@self.H@p <=0:
                t = solve_for_t(s,p,self.tr_radius)
                return s + t*p
            alpha = (r.T@r)/(p.T@self.H@p)
            s_new = s + alpha*p
            if np.linalg.norm(s_new) >= self.tr_radius:
                t = solve_for_t(s,p,self.tr_radius)
                return s + t*p
            r_new = r + alpha*(self.H@p)
            if np.linalg.norm(r_new) < epsilon*r_norm_0:
                return s_new
            beta = (r_new.T@r_new)/(r.T@r)
            p = -r_new + beta*p
            k += 1
            s = s_new
            r = r_new
        return s
    return tr_sol