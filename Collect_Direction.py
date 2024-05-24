import numpy as np
from icecream import ic

def Generate_Gaussian_Directions(x, num_direction):
    """
    Generate num_direction random directions for direct search
    Return num_direction normalized random directions 
    """
    x = np.array(x)
    dim = x.shape[0]
    directions = np.random.randn(num_direction, dim)
    return directions

def Orthogonalize(directions):
    """
    Orthogonalize a set of vectors using QR decomposition.
    """
    Q, R = np.linalg.qr(directions.T, mode='reduced')
    return Q.T

def Generate_Orthonormal_Directions(x, num_direction, old_directions):
    """
    Generate orthogonal directions based on old_directions using QR decomposition.
    """
    dim = x.shape[0]
    # Generate random directions
    new_matrix = np.random.randn(dim, num_direction)
    # Handle the case where old_directions is None
    if old_directions is None or old_directions.size == 0:
        orthogonal_directions = np.linalg.qr(new_matrix, mode='reduced')[0].T
    else:
        Q_old, _ = np.linalg.qr(old_directions.T, mode='reduced')
        # Calculate A - Q(Q^T A)
        new_proj = new_matrix - (Q_old @ Q_old.T) @ new_matrix
        Q_new, _ = np.linalg.qr(new_proj, mode='reduced')
        orthogonal_directions = Q_new.T
    return orthogonal_directions, Q_old.T


def Momentum(iter, history, n_m):
    """
    Collect_Momentum
    """
    old_x = history.iter_history[iter-1]['point']
    if iter < n_m:
        directions = Generate_Gaussian_Directions(old_x, n_m)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        return directions
       
    momentums = []
    for i in range(1,n_m+1):
        momentums.append(history.iter_history[iter-i]['point']-history.iter_history[iter-i-1]['point'])
    return np.array(momentums)

def Gradient(x,n_g, n_f, method_estimate_gradient,iter, history):
    """
    Here, we try to use SPSA that is used in MeZO.
    We can change the function SPSA to other gradient estimator methods.
    """
    # x_sample
    # history.record_results(x_sample.T, obj_values, iter, 'EG')
    if n_g == 0:
        return None
    obj_fun = history.obj_fun
    grad_stepsize = history.params[iter]['grad_stepsize']
    if method_estimate_gradient == 'Centered':
        x = np.array(x)
        z = Generate_Gaussian_Directions(x, n_g * n_f)
        x_samples = np.concatenate([x + grad_stepsize * z, x - grad_stepsize * z], axis=0)
        objs = np.array([obj_fun(x_sample) for x_sample in x_samples])
        
        history.record_results(x_samples, objs, iter, 'EG')
        
        grads = []
        for i in range(n_g):
            grad_sum = np.zeros_like(x)
            for j in range(n_f):
                idx = i * n_f + j
                obj1 = objs[idx]
                obj2 = objs[idx + n_g * n_f]
                grad_sum += (obj1 - obj2) / (2 * grad_stepsize) * z[idx]
            grads.append(grad_sum / n_f)
        return np.array(grads)
    else:
        pass

def DS_AND_ORTHOGONALIZE(x, n_ds_all, n_ds, directions, method_direct_search, iter, history):
    """
    Here, we first generate n_ds_all random directions for direct search
    And then we choose n_ds good directions and return
    """
    if n_ds_all == 0:
        return None
    # history.record_results(x_sample.T, obj_values, iter, 'DS')
    curr_obj = history.iter_history[iter-1]['objective']
    obj_fun = history.obj_fun
    ds_stepsize = history.params[iter]['ds_stepsize']
    if method_direct_search == 'Orthogonal':
        # ds_directions are orthonormal directions
        ds_directions, orthogonal_momentum_grad_directions = Generate_Orthonormal_Directions(x, n_ds_all, directions)
    else:
        pass
    x_samples = []
    objs = []
    ds_coordinates = []
    for direction in ds_directions:
        x_sample_plus = x + ds_stepsize * direction
        obj_val_plus = obj_fun(x_sample_plus)
        x_samples.append(x_sample_plus)
        objs.append(obj_val_plus)
        if obj_val_plus > curr_obj:
            # If the objective value of x_sample_plus is larger, sample along -z
            x_sample_minus = x - ds_stepsize * direction
            obj_val_minus = obj_fun(x_sample_minus)
            x_samples.append(x_sample_minus)
            objs.append(obj_val_minus)
            ds_coordinates.extend([1,-1])
        else:
            # Otherwise, sample a point with stepsize 2 * ds_stepsize along the same direction
            x_sample_far = x + 2 * ds_stepsize * direction
            obj_val_far = obj_fun(x_sample_far)
            x_samples.append(x_sample_far)
            objs.append(obj_val_far)
            ds_coordinates.extend([1,2])
    history.record_results(np.array(x_samples), np.array(objs), iter, 'DS')
    # Not using n_ds for now, return directions
    ds_coordinates = np.array(ds_coordinates)*ds_stepsize
    directions = np.concatenate((ds_directions, orthogonal_momentum_grad_directions), axis=0)
    return directions, ds_coordinates

def Collect_Direction(x, num_directions, method_collect_direction, iter, history):
    """
    history.params[iter] = {"tr_radius": tr_radius, "grad_stepsize": grad_stepsize, "ds_stepsize": ds_stepsize}
    """
    method_estimate_gradient = method_collect_direction['method_estimation_gradient']
    method_direct_search = method_collect_direction['method_direct_search']

    n_m, n_g, n_f, n_ds_all, n_ds = num_directions
    assert n_ds_all >= n_ds, "Error: n_ds_all is less than n_ds"

    # Collect the momentum direction
    momentums = Momentum(iter, history, n_m)

    # Collect the gradient direction
    if history.grad_fun is not None:
        gradient = history.evaluate_grad(x)
        gradient = np.atleast_2d(gradient)
    else:
        gradient = Gradient(x,n_g,n_f,method_estimate_gradient,iter,history)
    # Collect the direct search direction
    directions = np.empty((0, len(x)))
    directions = np.concatenate((directions,momentums), axis=0)
    directions = np.concatenate((directions,gradient), axis=0)
    # Add other directions based on direct search or other heuristics
    directions, ds_coordinates = DS_AND_ORTHOGONALIZE(x, n_ds_all, n_ds, directions, method_direct_search, iter, history)
    # sample n_m+n_g+n_s directions uniformly
    ic(iter, directions.shape)
    return directions, ds_coordinates

