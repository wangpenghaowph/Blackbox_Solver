from History_Manager import History_Manager
import Collect_Direction
import Solve_Subproblem

import numpy as np
class ARS:
    def __init__(
            self,
            x0, 
            obj_fun, 
            grad_fun, 
            method, #'Fixed_Dimension' 
            options,
        ):
        self.x = np.array(x0)
        self.n = len(x0)
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun if grad_fun else None
        self.history = History_Manager(x0,self.obj_fun, self.grad_fun)
        self.iter = 0
        self.method = method if method else 'Fixed_Dimension'
        self.options = options if options else {}
        self.collect_direction = Collect_Direction.Collect_Direction
        self.solve_subproblem = Solve_Subproblem.Solve_Subproblem
        self.success = 0
        self.configure()

    def configure(self):
        self.solver_tol = self.options.get('solver_tol', 1e-6)
        self.max_iter = self.options.get('max_iter', 100)
        self.tr_params = {
            'tr_init_radius': self.options.get('tr_init_radius', 1.0),
            'tr_min_radius': self.options.get('tr_min_radius', 1e-6),
            'tr_max_radius': self.options.get('tr_max_radius', 1e6),
            'tr_inc': self.options.get('tr_inc', 2.0),
            'tr_dec': self.options.get('tr_dec', 0.5),
            'tr_acceptance_threshold': self.options.get('tr_acceptance_threshold', [0.3, 0.7])
        }
        self.grad_params = {
            'grad_init_stepsize': self.options.get('grad_init_stepsize', 1e-6),
            'grad_min_stepsize': self.options.get('grad_min_stepsize', 1e-8),
            'grad_max_stepsize': self.options.get('grad_max_stepsize', 1e-2),
            'grad_inc': self.options.get('grad_inc', 2.0),
            'grad_dec': self.options.get('grad_dec', 0.5)
        }
        self.ds_params = {
            'ds_init_stepsize': self.options.get('ds_init_stepsize', 1),
            'ds_min_stepsize': self.options.get('ds_min_stepsize', 1e-5),
            'ds_max_stepsize': self.options.get('ds_max_stepsize', 1e5),
            'ds_inc': self.options.get('ds_inc', 2.0),
            'ds_dec': self.options.get('ds_dec', 0.5)
        }
        self.method_collect_direction = {
            'method_estimation_gradient': self.options.get('method_estimation_gradient', 'Centered'),
            'method_direct_search': self.options.get('method_direct_search', 'Uniform'),
        }
        self.method_solve_subproblem = {
            'method_subproblem': self.options.get('method_subproblem', 'PDFO'),
            'method_construct_model': self.options.get('method_construct_model', 'Quadratic'),
            'method_solve_tr': self.options.get('method_solve_tr', 'Dogleg'),
        }
        self.method_construct_model = self.options.get('method_construct_model', 'Quadratic')
        self.method_solve_tr = self.options.get('method_solve_tr', 'Dogleg')
        self.num_directions = self.options.get('num_directions', [1, 1, self.n, 0, 0])
        # initialize tr radius, grad stepsize, ds stepsize
        self.tr_radius = self.tr_params['tr_init_radius']
        self.grad_stepsize = self.grad_params['grad_init_stepsize']
        self.ds_stepsize = self.ds_params['ds_init_stepsize']
        # flag: EG, DS, CM, TR
        self.EG_flag = None
        self.DS_flag = None
        self.CM_flag = None
        self.TR_flag = None

        # status
        self.status = None
    def Check_Stop_Criteria(self):
        # check the stopping criteria. 
        # We are at iteration=self.iter, and we have the history of interation <= self.iter - 1
        if self.iter < 2:
            return False
        relative_diff = (self.history.iter_history[self.iter-2]['objective'] - self.history.iter_history[self.iter-1]['objective']) / max(1, abs(self.history.iter_history[self.iter-2]['objective']))
        if self.iter >= self.max_iter:
            self.status = 0
            return True
        elif relative_diff < self.solver_tol:
            self.status = 1
            return True
        else:
            return False
    # check and update tr radius
    def check_and_update_tr_radius(self, rho):
        """
        检查并更新信赖域半径

        参数:
        rho - 信赖域步长的接受率
        """
        if rho < self.tr_acceptance_threshold[0]:
            self.tr_flag = 0
            self.tr_radius = max(self.tr_params['tr_min_radius'], self.tr_radius * self.tr_params['tr_dec'])  
        elif rho > self.tr_acceptance_threshold[1]:
            self.tr_flag = 1
            self.tr_radius = min(self.tr_params['tr_max_radius'], self.tr_radius * self.tr_params['tr_inc'])
        #print(f"TR flag set to: {self.tr_flag}")
            
    # update grad stepsize
    def check_and_update_grad_stepsize(self):
        self.grad_stepsize = self.grad_stepsize
    # check ds flag and update ds stepsize
    def check_and_update_ds(self, ds_history):
        if ds_history is None:
            return
        obj_values = ds_history['objective']
        curr_value = self.history.iter_history[self.iter-1]['objective']
        smaller_than_curr = obj_values < curr_value
        proportion = np.sum(smaller_than_curr) / len(smaller_than_curr)
        if proportion == 0:
            self.ds_flag = 0
            self.ds_stepsize = max(self.ds_params['ds_min_stepsize'], self.ds_stepsize * self.ds_params['ds_dec'])
        elif proportion > 0: #TODO:add an option for this threshold
            self.ds_flag = 0
            self.ds_stepsize = min(self.ds_params['ds_max_stepsize'], self.ds_stepsize * self.ds_params['ds_inc'])
        #print(f"DS flag set to: {self.ds_flag}")
            
    def ARS_run(self):
        Solution = {}
        while not self.Check_Stop_Criteria():
            self.history.record_params(self.iter, self.tr_radius, self.grad_stepsize, self.ds_stepsize)
            
            directions = self.collect_direction(self.x, self.iter, self.num_directions, self.history)
            
            sol, obj = self.solve_subproblem(self.x, self.obj_fun, directions, self.method_solve_subproblem, self.iter, self.history)
            #TODO: update tr radius if using DRSOM method, therefore it is defined in Solve_Subproblem module(还没写update tr radius的代码,暂时最好不要用DRSOM方法#TODO:)
            best_entry = self.history.find_best_per_iter(self.iter)
            # update ds stepsize
            ds_history = self.history.total_history.get((self.iter,'DS'), None)
            self.check_and_update_ds(ds_history)
            # update grad stepsize
            self.check_and_update_grad_stepsize()
            self.x = best_entry['point']
            self.iter += 1
        Solution={
            'solution': self.history.iter_history[self.iter-1]['point'],
            'Objective': self.history.iter_history[self.iter-1]['objective'],
            'niter': self.iter,
            'nfev': self.history.get_nfev(),
            'pdfo_nfev': sum(self.history.pdfo_nfev),
            'ngrad': self.history.get_ngrad(),
            'status': self.status,
            'fhist': [self.history.iter_history[i]['objective'] for i in range(-1,len(self.history.iter_history)-1)],
        }
        return Solution, self.history