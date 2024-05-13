from History_Manager import History_Manager
import Sample
import Construct_Model
import Collect_Direction
import Solve_TR

class ARS:
    def __init__(
            self,
            x0, 
            obj_fun, 
            grad_fun=None, 
            method:chr=None, 
            options:dict=None,
            collect_direction=Collect_Direction.Collect_Direction,
            construct_model=Construct_Model.Construct_Model,
            solve_tr=Solve_TR.Solve_TR,
            ):
        self.x = x0
        self.n = len(x0)
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun
        self.history = History_Manager()
        self.iter = 0
        self.method = method if method else 'Fixed_Dimension'
        self.options = options if options else {}
        self.collect_direction = collect_direction 
        self.construct_model = construct_model
        self.solve_tr = solve_tr
        self.configure()

    def configure(self):
        self.tol = self.options.get('tol', 1e-4)
        self.max_iter = self.options.get('max_iter', 100)

        self.tr_init_radius = self.options.get('initial_trust_radius', 1.0)
        self.tr_radius = self.tr_init_radius
        self.tr_min_radius = self.options.get('tr_min_radius', 1e-6)
        self.tr_max_radius = self.options.get('tr_max_radius', 1e6)
        self.tr_inc = self.options.get('tr_inc', 2.0)
        self.tr_dec = self.options.get('tr_dec', 0.5)
        self.option_tr = {'tr_init_radius': self.tr_init_radius, 'tr_min_radius': self.tr_min_radius, 'tr_max_radius': self.tr_max_radius, 'tr_inc': self.tr_inc, 'tr_dec': self.tr_dec}
        
        self.grad_init_stepsize = self.options.get('grad_stepsize', 1e-6)
        self.grad_stepsize = self.grad_init_stepsize
        self.grad_min_stepsize = self.options.get('grad_min_stepsize', 1e-10)
        self.grad_max_stepsize = self.options.get('grad_max_stepsize', 1e-2)
        self.grad_inc = self.options.get('grad_inc', 2.0)
        self.grad_dec = self.options.get('grad_dec', 0.5)
        self.option_grad = {'grad_init_stepsize': self.grad_init_stepsize, 'grad_min_stepsize': self.grad_min_stepsize, 'grad_max_stepsize': self.grad_max_stepsize, 'grad_inc': self.grad_inc, 'grad_dec': self.grad_dec}

        self.ds_init_stepsize = self.options.get('ds_stepsize', 1e-6)
        self.ds_stepsize = self.ds_init_stepsize
        self.ds_min_stepsize = self.options.get('ds_min_stepsize', 1e-10)
        self.ds_max_stepsize = self.options.get('ds_max_stepsize', 1e-2)
        self.ds_inc = self.options.get('ds_inc', 2.0)
        self.ds_dec = self.options.get('ds_dec', 0.5)
        self.option_ds = {'ds_stepsize': self.ds_stepsize, 'ds_min_stepsize': self.ds_min_stepsize, 'ds_max_stepsize': self.ds_max_stepsize, 'ds_inc': self.ds_inc, 'ds_dec': self.ds_dec}

        self.method_construct_model = self.options.get('method_construct_model', 'Quadratic')
        self.method_direct_search = self.options.get('method_direct_search', 'Uniform')
        self.num_directions = self.options.get('num_directions', [1, 1, self.n, 0, 0])
        self.method_collect_direction = self.options.get('method_collect_direction', 'Collect_Direction')
        self.method_solve_tr = self.options.get('method_solve_tr', 'Dogleg')

        pass
    def Check_Stop_Criteria(self):
        # TODO: Implement the stopping criteria
        return False
    
    def Update_Params(self):
        # TODO:
        pass
    def ARS_run(self):
        while not self.Check_Stop_Criteria(self.history, self.iter, self.tol, self.max_iter):
            self.iter += 1
            directions = self.collect_direction(self.x, self.iter, self.num_directions, self.history)
            model = self.construct_model(self.x, directions, self.history, self.iter)
            tr_sol = self.solve_tr(model, self.x, self.history, self.iter)
            new_x = self.history.find_best_per_iter(self.iter)['point']
            self.Update_Params()
            self.x = new_x
        #TODO:
        return 1