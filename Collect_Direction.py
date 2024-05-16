import numpy as np

def SPSA(x, n_f, iter, history):
    """
    Estimate gradient by SPSA (The method used in MeZO)
    The n-SPSA gradient estimate averages gradient over n_f randomly sampled z
    Return gradient estimator 
    """
    x = np.array(x)
    obj_fun = history.obj_fun
    grad_stepsize = history.params[iter]['grad_stepsize']
    grads = []
    for _ in range(n_f):
        z = np.random.normal(0, 1, size=len(x))

        # First function evaluation
        x_sample1 = x + grad_stepsize * z
        obj1 = obj_fun(x_sample1)
        history.record_results(x_sample1, obj1, iter, 'EG')

        # Second function evaluation
        x_sample2 = x - grad_stepsize * z
        obj2 = obj_fun(x_sample2)
        history.record_results(x_sample2, obj2, iter, 'EG')
        grad = (obj1 - obj2) / (2 * grad_stepsize) * z
        grads.append(grad)

    grads = np.array(grads)
    return np.mean(grads,axis=0)



def Generate_Random_Directions(x, n_ds_all):
    """
    Generate n_ds_all random directions for direct search
    Return n_ds_all random directions 
    """
    x = np.array(x)
    dim = x.shape[0]
    directions = np.random.randn(n_ds_all, dim)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    return directions


def Choose_DS_Directions(directions,obj_values, n_ds):
    """
    Choose n_ds good directions
    """
    return directions



def Collect_Momentum(iter, history, n_m):
    """
    Collect_Momentum
    """
    if iter == 0 or iter == 1:
        return None
    
    momentums = []
    for i in range(1,n_m+1):
        if iter-i >= 1:
            momentums.append(history.iter_history[iter-i-1]['point']-history.iter_history[iter-i]['point'])
    return np.array(momentums)


def Collect_Gradient(x, iter, history, n_g, n_f):
    """
    Here, we try to use SPSA that is used in MeZO.
    We can change the function SPSA to other gradient estimator methods.
    """
    # x_sample
    # history.record_results(x_sample.T, obj_values, iter, 'EG')
    grads = []
    for _ in range(n_g):
        grad = SPSA(x, n_f, iter, history)
        grads.append(grad)
    return np.array(grads)


def Collect_DS(x, iter, history, n_ds_all, n_ds):
    """
    Here, we first generate n_ds_all random directions for direct search
    And then we choose n_ds good directions and return
    """

    # x_sample
    # history.record_results(x_sample.T, obj_values, iter, 'DS')
    obj_fun = history.obj_fun
    ds_stepsize = history.params[iter]['ds_stepsize']
    directions = Generate_Random_Directions(x, n_ds_all)
    x_sample = []
    obj_values = []
    for direction in directions:
        # positive direction
        x_sample1 = x + ds_stepsize * direction
        obj1 = obj_fun(x_sample1)
        x_sample.append(x_sample1)
        obj_values.append(obj1)

        # negative direction
        x_sample2 = x - ds_stepsize * direction
        obj2 = obj_fun(x_sample2)
        x_sample.append(x_sample2)
        obj_values.append(obj2)       
    
    x_sample = np.array(x_sample)
    obj_values = np.array(obj_values)
    history.record_results(x_sample, obj_values, iter, 'DS')

    # choose n_ds good directions
    directions = Choose_DS_Directions(directions,obj_values, n_ds)

    return directions

def Collect_Direction(x, iter, num_directions, history):
    """
    history.params[iter] = {"tr_radius": tr_radius, "grad_stepsize": grad_stepsize, "ds_stepsize": ds_stepsize}
    """
    n_m, n_g, n_f, n_ds_all, n_ds = num_directions
    assert n_ds_all >= n_ds, "Error: n_ds_all is less than n_ds"

    # Collect the momentum direction
    if n_m > 0:
        momentums = Collect_Momentum(iter, history, n_m)
    else:
        momentums = None

    # Collect the gradient direction
    if n_g > 0:
        if history.grad_fun is not None:
            gradient = history.evaluate_grad(x)
            gradient = np.atleast_2d(gradient)
        else:
        # Estimate Gradient
            gradient = Collect_Gradient(x,iter,history,n_g,n_f)
    else:
        gradient = None

    # Collect the direct search direction
    if n_ds_all > 0 and n_ds > 0:
        # TODO:
        # Add other directions based on direct search or other heuristics
        ds_direction = Collect_DS(x, iter, history, n_ds_all, n_ds)
    else:
        ds_direction = None

    # sample n_m+n_g+n_s directions uniformly
    directions = np.empty((0, len(x)))
    if isinstance(momentums, np.ndarray):
        directions = np.concatenate((directions,momentums), axis=0)
    if isinstance(gradient, np.ndarray):
        directions = np.concatenate((directions,gradient), axis=0)
    if isinstance(ds_direction, np.ndarray):
        directions = np.concatenate((directions,ds_direction), axis=0)
    
    assert isinstance(directions, np.ndarray), "Error: Don't have any useful direction"

    return directions

