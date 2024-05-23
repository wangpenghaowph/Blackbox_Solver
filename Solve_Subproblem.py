import numpy as np
from Construct_Model import Construct_Model
from Solve_TR import Solve_TR
from pdfo import pdfo
def Solve_Subproblem(x, obj_fun, directions, method_solve_subproblem, iter, history):
    if method_solve_subproblem['method_subproblem'] == 'PDFO':
        def sub_func(alpha):
            return obj_fun(x + np.dot(alpha,directions))
        alpha_0 = np.zeros(len(directions))
        options = {
            'maxiter': 500,
            'maxfev': 5000,
        }
        result = pdfo(sub_func, alpha_0, method='newuoa', options=options)
        subproblem_sol = result.x
        history.pdfo_nfev.append(result.nfev)
        history.pdfo_decrease.append(obj_fun(x)-result.fun)
    elif method_solve_subproblem['method_subproblem'] == 'PDFO_Single_Round':
        def sub_func(alpha):
            return obj_fun(x + np.dot(alpha,directions))
        alpha_0 = np.zeros(len(directions))
        options = {
            'maxiter': 1,
        }
        result = pdfo(sub_func, alpha_0, method='newuoa', options=options)
        subproblem_sol = result.x
        history.pdfo_nfev.append(result.nfev)
        history.pdfo_decrease.append(obj_fun(x)-result.fun)
    elif method_solve_subproblem['method_subproblem'] == 'DRSOM':
        #TODO: how to derive H and g
        model, n = Construct_Model(x, obj_fun, directions, method_solve_subproblem['method_construct_model'], iter, history)
        subproblem_sol = Solve_TR(model, n, history.params[iter]['tr_radius'], method_solve_subproblem['method_solve_tr'], H = None, g = None)
        # the original point is self.x + the linear combination of the directions with the coefficients in tr_sol 
        #if iter > 0:
            #rho = (self.history.iter_history[self.iter-1]['objective'] - obj) / (model(self.x) - model(best_entry['point']))
            #self.check_and_update_tr_radius(rho)
    sol = x + np.dot(subproblem_sol, directions)
    obj = obj_fun(sol)
    history.record_results(sol, obj, iter, 'TR')
    return sol, obj