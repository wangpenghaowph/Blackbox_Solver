import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def Construct_Model(x, obj_fun, directions, method_construct_model, iter, history):
    """
    # Input:
    x: the current point
    obj_fun: the objective function
    directions: a list of vectors in the original space e.g. [[1,2,3],[2,3,4],[3,4,5]]
    radius: trust region radius
    method_construct_model: method used to construct the model ('quadratic')
    iter: current iteration
    history: HistoryManager instance to record samples and objective values
    """
    radius = history.params[iter]['tr_radius']
    # Ensure x is a 2D numpy array
    x = np.asarray(x).reshape(-1, 1)
    directions = np.asarray(directions)
    n, m = directions.shape

    # Normalize directions
    norm_directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # Calculate the number of required samples
    num_samples_per_direction = math.ceil((m+1)*(m+2) / (2*m))

    # Generate sample points
    x_sample = np.hstack([x + (i + 1) * radius / (num_samples_per_direction) * norm_directions for i in range(num_samples_per_direction)])
    x_sample = np.hstack((x, x_sample))  # Ensure the first sample point is x

    # Evaluate objective function at sample points
    obj_values = np.apply_along_axis(obj_fun, 0, x_sample)

    # Record the sample points and their objective values
    history.record_results(x_sample.T, obj_values, iter, 'CM')

    def quadratic_model(x_new):
        # Construct quadratic model using polynomial features and linear regression
        projection_matrix = np.linalg.pinv(norm_directions)
        x_projected = np.dot(projection_matrix, x_new - x).flatten()
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform([x_projected])
        regressor = LinearRegression().fit(poly.fit_transform(x_projected.reshape(1, -1)), obj_values)
        #return regressor.predict(x_poly)[0]

    def model(x):
        return x[0]

    if method_construct_model == 'Quadratic':
        return model, n
    else:
        raise ValueError(f"Unknown method_construct_model: {method_construct_model}")

"""

# 示例目标函数
def obj_fun(x):
    return np.sum(x ** 2)

# 创建一个 HistoryManager 实例
class HistoryManager:
    def __init__(self, obj_fun, grad_fun=None):
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun
        self.total_history = {}
        self.iter_history = {}
        self.params = {}
        self.ngrad = 0

    def record_results(self, x, obj_values, iter, stage):
        if (iter, stage) not in self.total_history:
            self.total_history[(iter, stage)] = {
                "point": np.empty((0, x.shape[1])),
                "objective": np.empty(0),
                "stage": stage
            }
        self.total_history[(iter, stage)]["point"] = np.concatenate(
            (self.total_history[(iter, stage)]["point"], x), axis=0)
        self.total_history[(iter, stage)]["objective"] = np.concatenate(
            (self.total_history[(iter, stage)]["objective"], obj_values))
        return obj_values

# 测试构建模型函数
x = [1, 2, 3]
directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
method_construct_model = 'quadratic'
iter = 0
radius = 1.0

history = HistoryManager(obj_fun)
model, n = construct_model(x, obj_fun, directions, method_construct_model, iter, history, radius)

# 使用模型进行预测
x_new = np.array([2, 3, 4]).reshape(-1, 1)
predicted_value = model(x_new)
print("Predicted value:", predicted_value)
"""