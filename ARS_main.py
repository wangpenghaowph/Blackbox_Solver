from History_Manager import History_Manager
import Sample
import Construct_Model
import Collect_Direction
import Solve_TR
import numpy as np
class ARS:
    def __init__(
            self,
            x0, 
            obj_fun, 
            grad_fun, 
            method, #'Fixed_Dimension' 
            options,
            collect_direction=Collect_Direction.Collect_Direction,
            construct_model=Construct_Model.Construct_Model,
            solve_tr=Solve_TR.Solve_TR,
        ):
        self.x = x0
        self.n = len(x0)
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun if grad_fun else None
        self.history = History_Manager()
        self.iter = 0
        self.method = method if method else 'Fixed_Dimension'
        self.options = options if options else {}
        self.collect_direction = collect_direction 
        self.construct_model = construct_model
        self.solve_tr = solve_tr
        self.success = 0
        self.configure()

    def configure(self):
        self.solver_tol = self.options.get('solver_tol', 1e-6)
        self.max_iter = self.options.get('max_iter', 100)
        # trust region params
        self.tr_init_radius = self.options.get('initial_trust_radius', 1.0)
        self.tr_radius = self.tr_init_radius
        self.tr_min_radius = self.options.get('tr_min_radius', 1e-6)
        self.tr_max_radius = self.options.get('tr_max_radius', 1e6)
        self.tr_inc = self.options.get('tr_inc', 2.0)
        self.tr_dec = self.options.get('tr_dec', 0.5)
        self.option_tr = {'tr_init_radius': self.tr_init_radius, 'tr_min_radius': self.tr_min_radius, 'tr_max_radius': self.tr_max_radius, 'tr_inc': self.tr_inc, 'tr_dec': self.tr_dec}
        # grad params
        self.grad_init_stepsize = self.options.get('grad_stepsize', 1e-6)
        self.grad_stepsize = self.grad_init_stepsize
        self.grad_min_stepsize = self.options.get('grad_min_stepsize', 1e-8)
        self.grad_max_stepsize = self.options.get('grad_max_stepsize', 1e-2)
        self.grad_inc = self.options.get('grad_inc', 2.0)
        self.grad_dec = self.options.get('grad_dec', 0.5)
        self.option_grad = {'grad_init_stepsize': self.grad_init_stepsize, 'grad_min_stepsize': self.grad_min_stepsize, 'grad_max_stepsize': self.grad_max_stepsize, 'grad_inc': self.grad_inc, 'grad_dec': self.grad_dec}
        # direct search params
        self.ds_init_stepsize = self.options.get('ds_stepsize', 1e-6)
        self.ds_stepsize = self.ds_init_stepsize
        self.ds_min_stepsize = self.options.get('ds_min_stepsize', 1e-10)
        self.ds_max_stepsize = self.options.get('ds_max_stepsize', 1e-2)
        self.ds_inc = self.options.get('ds_inc', 2.0)
        self.ds_dec = self.options.get('ds_dec', 0.5)
        self.option_ds = {'ds_stepsize': self.ds_stepsize, 'ds_min_stepsize': self.ds_min_stepsize, 'ds_max_stepsize': self.ds_max_stepsize, 'ds_inc': self.ds_inc, 'ds_dec': self.ds_dec}

        self.method_collect_direction = self.options.get('method_collect_direction', 'Centered_Uniform')
        self.method_construct_model = self.options.get('method_construct_model', 'Quadratic')
        self.method_direct_search = self.options.get('method_direct_search', 'Uniform')
        self.num_directions = self.options.get('num_directions', [1, 1, self.n, 0, 0])
        self.method_collect_direction = self.options.get('method_collect_direction', 'Collect_Direction')
        self.method_solve_tr = self.options.get('method_solve_tr', 'Dogleg')

        self.tr_acceptance_threshold = self.options.get('tr_acceptance_threshold', [0.3, 0.7])
        self.threshold=[]

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
        if rho < self.tr_acceptance_threshold[0]:
            self.tr_flag = 0
            self.tr_radius = max(self.tr_min_radius, self.tr_radius * self.tr_dec)  
        elif rho > self.tr_acceptance_threshold[1]:
            self.tr_flag = 1
            self.tr_radius = min(self.tr_max_radius, self.tr_radius * self.tr_inc)
        #print(f"TR flag set to: {self.tr_flag}")
            
    # update grad stepsize
    def check_and_update_grad_stepsize(self, obj_values):
        self.grad_stepsize = self.grad_stepsize
    # check ds flag and update ds stepsize
    def check_and_update_ds(self, obj_values):
        curr_obj = obj_values[0]
        smaller_than_first = obj_values[1:] < curr_obj
        proportion = np.sum(smaller_than_first) / len(smaller_than_first)
        if proportion == 0:
            self.ds_flag = 0
            self.ds_stepsize = max(self.ds_min_stepsize, self.ds_stepsize * self.ds_dec)
        elif proportion > 0.3: #TODO:add an option for this threshold
            self.ds_flag = 0
            self.ds_stepsize = min(self.ds_max_stepsize, self.ds_stepsize * self.ds_inc)
        #print(f"DS flag set to: {self.ds_flag}")
            
    def ARS_run(self):
        Solution = {}
        while not self.Check_Stop_Criteria():
            self.history.record_params(self.iter, self.tr_radius, self.grad_stepsize, self.ds_stepsize)
            directions = self.collect_direction(self.x, self.iter, self.num_directions, self.history)
            model, n = self.construct_model(self.x, self.obj_fun, directions, self.method_construct_model, self.iter, self.history)
            tr_sol = self.solve_tr(model, n, self.tr_radius, self.method_solve_tr)
            # the original point is self.x + the linear combination of the directions with the coefficients in tr_sol
            # self.x is a np.array, tr_sol['point'] is a np.array, directions is a np.array with each element is a np.array
            tr_original_sol = self.x + np.dot(tr_sol, directions)
            tr_obj = self.obj_fun(tr_original_sol)
            self.history.record_results(tr_original_sol, tr_obj, self.iter, 'TR')
            best_entry = self.history.find_best_per_iter(self.iter)['point']
            # update ds stepsize
            self.check_and_update_ds(self.history.total_history[(self.iter,'DS')]['objective'])
            # update tr radius
            rho = (self.history.iter_history[self.iter-1]['objective'] - tr_obj) / (model(self.x) - model(best_entry['point']))
            self.check_and_update_tr_radius(rho)
            # update grad stepsize
            self.update_and_update_grad_stepsize()

            self.x = best_entry['point']
            self.iter += 1

        Solution={
            'solution': self.history.iter_history[-1]['point'],
            'Objective': self.history.iter_history[-1]['objective'],
            'niter': self.iter,
            'nfev': self.history.get_nfev,
            'ngrad': self.history.get_ngrad,
            'status': self.status,
            'fhist': [self.history.iter_history[i]['objective'] for i in range(len(self.history.iter_history))],
        }
        return Solution