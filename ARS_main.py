from History_Manager import History_Manager
import Sample
import Construct_Model
import Collect_Direction
import Solve_TR

class ARS:
    def __init__(self, x0, obj_fun, grad_fun=None, method:chr=None, options:dict=None):
        self.x = x0
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun
        self.history = History_Manager()
        self.iter = 0
        self.method = method if method else {}
        self.options = options if options else {}
    def fill_options(self):
        self.tol = self.options.get('tol', 1e-4)
        self.max_iter = self.options.get('max_iter', 100)
        self.tr_init_radius = self.options.get('initial_trust_radius', 1.0)
        self.tr_min_radius = self.options.get('tr_min_radius', 1e-6)
        self.tr_max_radius = self.options.get('tr_max_radius', 1e6)
        self.tr_inc = self.options.get('tr_inc', 2.0)
        self.tr_dec = self.options.get('tr_dec', 0.5)
        self.grad_init_stepsize = self.options.get('grad_step_size', 1e-6)
        self.grad_min_stepsize = self.options.get('grad_min_step_size', 1e-10)
        self.grad_max_stepsize = self.options.get('grad_max_step_size', 1e-2)
        self.grad_inc = self.options.get('grad_inc', 2.0)
        self.grad_dec = self.options.get('grad_dec', 0.5)
        self.ds_stepsize = self.options.get('ds_step_size', 1e-6)
        self.ds_min_stepsize = self.options.get('ds_min_step_size', 1e-10)
        self.ds_max_stepsize = self.options.get('ds_max_step_size', 1e-2)
        self.ds_inc = self.options.get('ds_inc', 2.0)
        self.ds_dec = self.options.get('ds_dec', 0.5)
        self.method_construct_model = self.options.get('method_construct_model', 'Quadratic')
        self.method_direct_search = self.options.get('method_direct_search', 'Uniform')
        #TODO:self.method_trust_region = self.options.get('method_trust_region', 'Dogleg')

        pass
    def Check_Stop_Criteria(self):
        # TODO: Implement the stopping criteria
        return False
    
    def ARS_run(self):
        while not self.Check_Stop_Criteria():
            self.iter += 1
            directions = Collect_Direction.Collect_Direction(self.x, self.grad_fun, self.history, self.iter)
            model = Construct_Model.Construct_Model(self.x, directions, self.history, self.iter)
            tr_sol = Solve_TR.Solve_TR(model, self.x, self.history, self.iter)
            # Update solution based on evaluation of the objective function
            new_x = self.history.find_best_per_iter(self.iter)['point']
            # Decision to accept new_x, update self.x and self.obj, and adjust trust radius
            # TODO: Implement the logic to update x and obj based on new_x and new_obj
            self.x = new_x
            