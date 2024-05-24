import numpy as np
from pdfo import pdfo
from icecream import ic
from scipy import linalg as LA
import numpy as np
from scipy.sparse import csr_matrix
from functools import reduce

def Solve_Subproblem(x, obj_fun, directions, ds_coordinates, method_solve_subproblem, iter, history):
    """
    directions are already orthonormal
    ds_coordinates is a numpy array of length 2*k, where k is the number of DS directions
    The first k directions are DS directions
    """
    sub_dim = len(directions)
    tr_radius = history.params[iter]['tr_radius']
    ic(iter, sub_dim, tr_radius)
    if method_solve_subproblem['method_subproblem'] == 'PDFO':
        def sub_func(alpha):
            return obj_fun(x + np.dot(alpha,directions))
        alpha_0 = np.zeros(sub_dim)
        options = {
            'radius_init': tr_radius,
            'maxfev': 5000,
        }
        result = pdfo(sub_func, alpha_0, method='newuoa', options=options)
        subproblem_sol = result.x
        history.pdfo_nfev.append(result.nfev)
        history.pdfo_decrease.append(obj_fun(x)-result.fun)
        sol = x + np.dot(subproblem_sol, directions)
        obj = obj_fun(sol)
        history.record_results(sol, obj, iter, 'TR')
        return None
    elif method_solve_subproblem['method_subproblem'] == 'PDFO_Single_Round':
        def sub_func(alpha):
            return obj_fun(x + np.dot(alpha,directions))
        alpha_0 = np.zeros(len(directions))
        options = {
            'radius_init': tr_radius,
            'maxfev': 10 * sub_dim,
        }
        result = pdfo(sub_func, alpha_0, method='newuoa', options=options)
        subproblem_sol = result.x
        history.pdfo_nfev.append(result.nfev)
        history.pdfo_decrease.append(obj_fun(x)-result.fun)
        sol = x + np.dot(subproblem_sol, directions)
        obj = obj_fun(sol)
        history.record_results(sol, obj, iter, 'TR')
        return None
    elif method_solve_subproblem['method_subproblem'] == 'DRSOM':
        g, H = Construct_Model(x, obj_fun, directions, ds_coordinates, method_solve_subproblem['method_construct_model'], iter, history)
        tr_sol, predicted_reduced = Solve_TR(g, H, tr_radius)
        # the original point is self.x + the linear combination of the directions with the coefficients in tr_sol 
        sol = x + tr_sol @ directions
        obj = obj_fun(sol)
        rho = (history.iter_history[iter-1]['objective'] - obj) / predicted_reduced
        return rho
    elif method_solve_subproblem['method_subproblem'] == 'Skip':
        pass

def Construct_Model(x, obj_fun, directions, ds_coordinates, method_construct_model, iter, history):
    if method_construct_model == 'Quadratic':
        curr_obj = history.iter_history[iter-1]['objective']
        tr_radius = history.params[iter]['tr_radius']
        n_total_directions = len(directions)
        n_ds_directions = len(ds_coordinates) // 2
        other_direction_samples = []
        other_objs = []
        other_coordinates = []

        ic('Sample other points')

        for direction in directions[n_ds_directions:]:
            x_sample_plus = x + tr_radius * direction
            obj_val_plus = obj_fun(x_sample_plus)
            other_direction_samples.append(x_sample_plus)
            other_objs.append(obj_val_plus)
            if obj_val_plus > curr_obj:
                # If the objective value of x_sample_plus is larger, sample along -z
                x_sample_minus = x - tr_radius * direction
                obj_val_minus = obj_fun(x_sample_minus)
                other_direction_samples.append(x_sample_minus)
                other_objs.append(obj_val_minus)
                other_coordinates.extend([tr_radius,-tr_radius])
            else:
                # Otherwise, sample a point with stepsize 2 * ds_stepsize along the same direction
                x_sample_far = x + 2 * tr_radius * direction
                obj_val_far = obj_fun(x_sample_far)
                other_direction_samples.append(x_sample_far)
                other_objs.append(obj_val_far)
                other_coordinates.extend([tr_radius,2*tr_radius])
        history.record_results(np.array(other_direction_samples), np.array(other_objs), iter, 'CM')
        ds_objs = history.total_history[(iter, 'DS')]['objective']
        ic(len(ds_objs), len(ds_coordinates))
        total_coordinates = np.concatenate((ds_coordinates, other_coordinates))
        total_objs = np.concatenate((ds_objs, other_objs))

        # Initialize g and H
        g = np.zeros(n_total_directions)
        H = np.zeros((n_total_directions, n_total_directions))
        # Calculate g and H using the quadratic interpolation
        obj_better = []
        step_better = []
        for i in range(n_total_directions):
            f1 = total_objs[2*i]
            f2 = total_objs[2*i+1]
            step1 = total_coordinates[2*i]
            step2 = total_coordinates[2*i+1]
            if f1 < f2:
                obj_better.append(f1)
                step_better.append(step1)
            else:
                obj_better.append(f2)
                step_better.append(step2)
            if step2 < 0:
                # then the sampling is symmetric, i.e. step1 = -step2
                g[i] = (f1 - f2) / (2 * step1) 
                H[i, i] = (f1 + f2 - 2 * curr_obj) / (step1**2)
            else:
                g[i] = (4 * f1 - f2 - 3 * curr_obj) / (step1 - step2)
                H[i, i] = (-2 * f1 + f2 + curr_obj) / (step1**2)

        ic('sample cross points')

        cross_samples = []
        cross_objs = []
        for i in range(n_total_directions):
            for j in range(i+1, n_total_directions):
                cross_sample = x + step_better[i] * directions[i] + step_better[j] * directions[j]
                cross_obj = obj_fun(cross_sample)
                cross_samples.append(cross_sample)
                cross_objs.append(cross_obj)
                H[i, j] = (cross_obj - obj_better[i] - obj_better[j] + curr_obj) / (step_better[i] * step_better[j])
        history.record_results(np.array(cross_samples), np.array(cross_objs), iter, 'CM')
        return g, H

def solve_for_t(s, p, tr_radius):
    c_coef = s.T @ s - tr_radius**2
    b_coef = 2 * s.T @ p
    a_coef = p.T @ p
    # Calculate the discriminant
    discriminant = b_coef**2 - 4 * a_coef * c_coef

    if discriminant < 0:
        raise ValueError("No real solution exists for the given c.")

    # Calculate the two possible values of t
    t = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
    return t

def Solve_TR(g, H, tr_radius):
    epsilon = 1e-4
    sub_dim = len(g)
    max_iter = sub_dim
    s = np.zeros(sub_dim)
    r = g
    r_norm_0 = np.linalg.norm(r)
    p = -g
    k = 0
    if r_norm_0 < epsilon:
        return s, 0
    while k < max_iter:
        pTHp = p.T @ H @ p
        if pTHp <= 0:
            t = solve_for_t(s, p, tr_radius)
            s = s + t * p
            return s, g.T @ s + 0.5 * s.T @ H @ s
        alpha = (r.T@r)/(p.T@H@p)
        s_new = s + alpha * p
        if np.linalg.norm(s_new) >= tr_radius:
            t = solve_for_t(s, p, tr_radius)
            s = s + t * p
            return s, g.T @ s + 0.5 * s.T @ H @ s
        r_new = r + alpha * H @ p
        if np.linalg.norm(r_new) < epsilon * r_norm_0:
            return s_new, g.T @ s_new + 0.5 * s_new.T @ H @ s_new
        beta = (r_new.T@r_new)/(r.T@r)
        p = -r_new + beta * p
        k += 1
        s = s_new
        r = r_new
    return s, g.T @ s + 0.5 * s.T @ H @ s





"""
The following is to solve trust region subproblem

Copy from https://github.com/PengchengXieLSEC/Large-Scale-Python/blob/main/python/trust_sub.py
"""
    
def _secular_eqn(lambda_0, eigval, alpha, delta):
    """
    The function secular_eqn returns the value of the secular
    equation at a set of m points.
    """
    m = lambda_0.size
    n = len(eigval)
    unn = np.ones((n, 1))
    unm = np.ones((m, 1))
    M = np.dot(eigval, unm.T) + np.dot(unn, lambda_0.T)
    MC = M.copy()
    MM = np.dot(alpha, unm.T)
    M[M != 0.0] = MM[M != 0.0] / M[M != 0.0]
    M[MC == 0.0] = np.inf * np.ones(MC[MC == 0.0].size)
    M = M*M
    value = np.sqrt(unm / np.dot(M.T, unn))

    if len(value[np.where(value == np.inf)]):
        inf_arg = np.where(value == np.inf)
        value[inf_arg] = np.zeros((len(value[inf_arg]), 1))

    value = (1.0/delta) * unm - value

    return value

def rfzero(x, itbnd, eigval, alpha, delta, tol):
    """
    This function finds the zero of a function
    of one variable to the RIGHT of the starting point x.
    The code contanins a small modification of the M-file fzero in matlab,
    to ensure a zero to the right of x is searched for.
    """
    # start the iteration counter
    itfun = 0

    # find the first three points, a, b, and c and their values
    if (x != 0.0):
        dx = abs(x) / 2
    else:
        dx = 0.5

    a = x
    c = a
    fa = _secular_eqn(a, eigval, alpha, delta)
    itfun = itfun + 1

    b = x + dx
    fb = _secular_eqn(b, eigval, alpha, delta)
    itfun = itfun + 1

    # find change of sign
    while ((fa > 0) == (fb > 0)):

        dx = 2*dx

        if ((fa > 0) != (fb > 0)):
            break
        b = x + dx
        fb = _secular_eqn(b, eigval, alpha, delta)
        itfun = itfun + 1

        if (itfun > itbnd):
            break

    fc = fb

    # main loop, exit from the middle of the loop
    while (fb != 0):
        # Ensure that b is the best result so far, a is the previous
        # value of b, and c is on hte opposit side of 0 from b
        if (fb > 0) == (fc > 0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # convergence test and possible exit
        if itfun > itbnd:
            break

        m = 0.5 * (c-b)
        rel_tol = 2.0 * tol * max(abs(b), 1.0)

        if (abs(m) <= rel_tol) or (abs(fb) < tol):
            break

        # choose bisection or interpolation
        if (abs(e) < rel_tol) or (abs(fa) <= abs(fb)):
            # bisection
            d = e = m
        else:
            # interpolation
            s = float(fb)/fa
            if a == c:
                # linear interpolation
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quad interpolation
                q = float(fa)/fc
                r = float(fb)/fc
                p = s * (2.0 * m * q * (q-r) - (b-a) * (r-1.0))
                q = (q-1.0) * (r-1.0) * (s-1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            # Check if the interpolated point is acceptable
            if (2.0*p < 3.0*m*q - abs(rel_tol*q)) and (p < abs(0.5*e*q)):
                e = d
                d = float(p)/q
            else:
                d = m
                e = m
            #  end of iterpolation

        # Next point
        a = b
        fa = fb
        if (abs(d) > rel_tol):
            b = b + d
        else:
            if b > c:
                b = b - rel_tol
            else:
                b = b + rel_tol

        fb = _secular_eqn(b, eigval, alpha, delta)
        itfun = itfun + 1

    return (b, c, itfun)


def trust_sub(g, H, delta):
    """
    This function solves the trust region subproblem when the
    Frobenuis norm of H is not very small.
    The subproblem is:
        min g.T s + 1/2 s.T H s
        s.t. || s || <= delta

    Note that any restriction that the problem has
    can be added to the constriants in the trust region.
    In that case the following algorithm will not work and
    another package should be used. The alternative is to
    penalize the constraints violations in the objective function
    evaluations.
    """

    tol = 10e-12
    tol_seqeq = 10e-8
    key = 0
    itbnd = 50
    lambda_0 = 0
    s_factor = 0.8
    b_factor = 1.2
    n = len(g)
    coeff = np.zeros((n, 1))

    # convert H to full matrix if sparse
    T = csr_matrix(H)
    T = T.todense()
    H = np.squeeze(np.asarray(T))

    # get the eigen value and vector
    D, V = LA.eigh(0.5 * (H.T + H))
    count = 0
    eigval = D[np.newaxis].T
    # find the minimum eigen value
    jmin = np.argmin(eigval)
    mineig = np.amin(eigval)

    # depending on H, find a step size
    alpha = np.dot(-V.T, g)
    sig = (np.sign(alpha[jmin]) + (alpha[jmin] == 0).sum())[0]

    # PSD case
    if mineig > 0:
        lambda_0 = 0
        coeff = alpha * (1/eigval)
        s = np.dot(V, coeff)
        # That is, s = -v (-v.T g./eigval)
        nrms = LA.norm(s)
        if nrms < b_factor*delta:
            key = 1
        else:
            laminit = np.array([[0]])
    else:
        laminit = -mineig

    # Indefinite case
    if key == 0:
        if _secular_eqn(laminit, eigval, alpha, delta) > 0:
          b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)

          if abs(_secular_eqn(b, eigval, alpha, delta)) <= tol_seqeq:
              lambda_0 = b
              key = 2
              lam = lambda_0 * np.ones((n, 1))

              coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

              if (nrms > b_factor * delta or nrms < s_factor * delta):
                  key = 5
                  lambda_0 = -mineig
          else:
                key = 3
                lambda_0 = -mineig
        else:
            key = 4
            lambda_0 = -mineig

        lam = lambda_0 * np.ones((n, 1))

        if key > 2:
            arg = abs(eigval + lam) < 10 * (np.finfo(float).eps *
                np.maximum(abs(eigval), np.ones((n,1))))
            alpha[arg] = 0.0

        coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

        if key > 2 and nrms < s_factor * delta:
            beta = np.sqrt(delta**2 - nrms**2)
            s = s + reduce(np.dot, [beta, sig, V[:, jmin]]).reshape(n, 1)

        if key > 2 and nrms > b_factor * delta:
            b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
            lambda_0 = b
            lam = lambda_0 * np.ones((n, 1))

            coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

    # return the model prediction of the change in the objective with s
    val = np.dot(g.T, s) + reduce(np.dot, [(.5*s).T, H, s])

    return (s, val)

def compute_step(alpha, eigval, coeff, V, lam):
    w = eigval + lam
    arg1 = np.logical_and(w == 0, alpha == 0)
    arg2 = np.logical_and(w == 0, alpha != 0)
    coeff[w != 0] = alpha[w != 0] / w[w != 0]
    coeff[arg1] = 0
    coeff[arg2] = np.inf
    coeff[np.isnan(coeff)] = 0
    s = np.dot(V, coeff)
    nrms = LA.norm(s)
    return(coeff, s, nrms, w)