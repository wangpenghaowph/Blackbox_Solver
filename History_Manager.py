# This module records all the necessary history information
# History information is of the form:
# {iter, stage} -> {point, objective, stage}

# Any function that changes the history information should be defined in this module
"""
Functions in this module:
"""
import numpy as np

class History_Manager:
    def __init__(self,starting_point, obj_fun, grad_fun):
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun if grad_fun else None
        self.total_history = {}
        self.iter_history = {-1:{"point": starting_point, "objective": self.obj_fun(starting_point), "stage": 'init'}}
        self.pdfo_nfev = []
        self.params = {}
        self.ngrad = 0
    # stage is in ['EG',DS','CM','TR'] (Estimate Gradient, Direct Search, Construct Model, TR_sol)
    def evaluate(self, x, iter, stage):
        # 把x转化为嵌套向量
        x = np.atleast_2d(x)
        obj_values = [self.obj_fun(xi) for xi in x]
        return obj_values
    
    def record_results(self, x, obj_values, iter, stage):
        # 把x转化为嵌套向量
        x = np.atleast_2d(x)
        obj_values = np.atleast_1d(obj_values)
        
        if (iter, stage) not in self.total_history:
            self.total_history[(iter, stage)] = {
                "point": np.empty((0, x.shape[1])),  # 初始化为空的2D数组
                "objective": np.empty(0),  # 初始化为空的1D数组
                "stage": stage
            }
        
        # 使用 numpy 的 concatenate 方法来扩展数组
        self.total_history[(iter, stage)]["point"] = np.concatenate(
            (self.total_history[(iter, stage)]["point"], x), axis=0)
        self.total_history[(iter, stage)]["objective"] = np.concatenate(
            (self.total_history[(iter, stage)]["objective"], obj_values))
        
    def record_params(self, iter, tr_radius, grad_stepsize, ds_stepsize):
        self.params[iter] = {"tr_radius": tr_radius, "grad_stepsize": grad_stepsize, "ds_stepsize": ds_stepsize}

    # Use get_best_iterate after solving the TR problem. To get the best iterate in each iter.
    # return: {"point": a numpy array, "objective": a scalar, "stage": stage}
    def find_best_per_iter(self, iter):
        stages = ['EG', 'DS', 'CM', 'TR']
        best_entry = None
        
        for stage in stages:
            key = (iter, stage)
            if key in self.total_history:
                entries = self.total_history[key]
                objectives = entries["objective"]
                points = entries["point"]
                
                if objectives.size > 0:
                    min_index = np.argmin(objectives)
                    current_best = {"point": points[min_index], "objective": objectives[min_index], "stage": stage}
                    
                    if best_entry is None or current_best["objective"] < best_entry["objective"]:
                        best_entry = current_best
        
        self.iter_history[iter] = best_entry
        return best_entry
        
    def evaluate_grad(self, x):
        if self.grad_fun:
            grad = self.grad_fun(x)
            self.ngrad += 1
            return grad
        else:
            raise ValueError("No gradient function provided")
        
    def get_nfev(self):
        # 返回 total_history 中所有不同的 x 点的数量
        distinct_x = set()
        for key in self.total_history:
            for point in self.total_history[key]["point"]:
                distinct_x.add(tuple(point))
        return len(distinct_x)
    def get_ngrad(self):
        return self.ngrad
    