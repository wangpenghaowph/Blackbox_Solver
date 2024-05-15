import numpy as np

def Collect_Momentum(iter, n_m,history):
    #TODO:record number of 
    """
    history.iter_history[-1]['point],[-2],[-3] 
    """
    pass
def Collect_Gradient(x, iter, history):
    # x_sample
    # history.record_results(x_sample.T, obj_values, iter, 'EG')
    pass
def Collect_DS(x, iter, history):
    # x_sample
    # history.record_results(x_sample.T, obj_values, iter, 'DS')

    # 
    pass    
def Collect_Direction(x, iter, num_directions, history):
    """
    history.params[iter] = {"tr_radius": tr_radius, "grad_stepsize": grad_stepsize, "ds_stepsize": ds_stepsize}
    """
    n_m, n_g, n_f, n_ds_all, n_ds = num_directions
    directions = []
    # Collect the momentum direction
    if n_m > 0:
        #momentums =
        #directions['momentum'] = [momentums]
        pass
    if history.grad_fun is not None:
        gradient = history.evaluate_grad(x)
        # Store the gradient as one of the directions
        directions
    else:
        pass
        # TODO: Estimate Gradient
    if n_ds > 0:
        pass
        # TODO:
        # Add other directions based on direct search or other heuristics
    # sample n_m+n_g+n_s directions uniformly
    directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return directions

def generate_random_directions(x, n_all, ds):
    # 确保x是numpy数组
    x = np.array(x)
    # 获取x的维度
    dim = x.shape[0]
    # 生成n_all个随机方向
    directions = np.random.randn(n_all, dim)
    # 将方向向量标准化为单位向量
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    # 生成2n_all个点，分别在正负方向上移动步长ds
    points = []
    for direction in directions:
        points.append(x + ds * direction)
        points.append(x - ds * direction)
    return np.array(points)
