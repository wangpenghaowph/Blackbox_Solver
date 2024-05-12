# This module records all the necessary history information

class History_Manager:
    def __init__(self, obj_fun, grad_fun=None):
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun
        self.nfev = 0
        self.ngrad = 0
        self.total_history = {}
        self.iter_history = {}
    # stage is in ['EG',DS','TR'] (Estimate Gradient, Direct Search, Trust Region)
    def evaluate(self, x, obj_fun, iter, stage):
        if x and isinstance(x[0], list):
            # 假设x是嵌套列表，每个元素代表一个向量
            obj_values = [obj_fun(xi) for xi in x]
        else:
            # 假设x是单个向量
            obj_values = [obj_fun(x)]
            x=[x]
        self.nfev += len(x)
        self.total_history[(iter, stage)] = {"point": x, "objective": obj_values, "stage": stage}
        return obj_values
    # Use get_best_iterate after solving the TR problem. To get the best iterate in each iteration.
    def find_best_per_iter(self, iter):
        # 假设已知的 stage 类型
        stages = ['EG', 'DS', 'TR']
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
            self.iter_history[iter] = best_entry
        return best_entry
    def evaluate_grad(self, x, grad_fun, iter, stage):
        self.ngrad += 1
        grad = grad_fun(x)
        self.total_history[(iter, stage)]["gradient"] = grad

    def get_nfev(self):
        return self.nfev

    def get_ngrad(self):
        return self.ngrad
