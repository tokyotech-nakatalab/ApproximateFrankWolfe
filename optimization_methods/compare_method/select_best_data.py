from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from optimization_methods.base_optimization_method import *

class SelectBestData(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        self.best_x = init_x
        self.best_true_obj = self.problem.objective_true(self.best_x)
        ave_rho, ave_diameter, ave_rdm = -1, -1, -1
        return self.best_x, -1, self.best_true_obj, ave_rho, ave_diameter, ave_rdm
