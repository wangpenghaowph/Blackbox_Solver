# This module records all the necessary history information
# History information is of the form:
# {iter, stage} -> {point, objective, stage}

# Any function that changes the history information should be defined in this module
"""
Functions in this module:
"""
class History_Manager:
    def __init__(self, obj_fun, grad_fun=None):
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun if grad_fun else None
        self.total_history = {}
        self.iter_obj_history = {}
        self.params = {}
        self.ngrad = 0
    # stage is in ['EG',DS','CM','TR'] (Estimate Gradient, Direct Search, #TODO:Trust Region solution)
    def evaluate(self, x, iter, stage):
        if x and isinstance(x[0], list):
            # 假设x是嵌套列表，每个元素代表一个向量
            obj_values = [self.obj_fun(xi) for xi in x]
        else:
            # 假设x是单个向量
            obj_values = [self.obj_fun(x)]
            x=[x]
        self.total_history[(iter, stage)] = {"point": x, "objective": obj_values, "stage": stage}
        return obj_values
    # Use get_best_iterate after solving the TR problem. To get the best iterate in each iteration.
    def find_best_per_iter(self, iter):
        # 假设已知的 stage 类型
        stages = ['EG', 'DS', 'CM', 'TR']
        best_entry = None
        # 直接遍历已知的 stage
        for stage in stages:
            key = (iter, stage)
            if key in self.total_history:
                entries = self.total_history[key]
                current_best = min(entries, key=lambda entry: entry["objective"], default=None)
                if current_best is not None:
                    if best_entry is None or current_best["objective"] < best_entry["objective"]:
                        best_entry = current_best
                        best_entry['stage'] = stage  # 记录此条目的 stage
        # 如果找到最佳条目，则存储并返回它
        if best_entry:
            self.iter_obj_history[iter] = best_entry
        return best_entry
    def evaluate_grad(self, x):
        grad = self.grad_fun(x) if self.grad_fun else print("No gradient function provided")
        self.ngrad += 1
        return grad

    def get_nfev(self):
        # return the distinct x in total_history
        # find the distinct x in total_history
        distinct_x = set()
        for key in self.total_history:
            distinct_x.update(self.total_history[key]["point"])

    def get_ngrad(self):
        return self.ngrad
    
       ## self.x_history.extend(x)
        #self.obj_history.extend([(obj, iter, stage) for obj in obj_values])
        
        #return obj_values
