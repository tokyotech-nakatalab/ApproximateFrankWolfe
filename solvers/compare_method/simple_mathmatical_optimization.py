from utility.setting import *
from solvers.base_optimization_method import *
from utility.module import environment

class MathmaticalOptimization(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        for i in range(g.n_item):
            fs[i].set_parameter_for_opt(new_s[i])
        if environment == TSUBAME:
            opt_x, opt_value = self.problem.modlize_casadi_problem(fs, new_s, prev_x=init_x)
        elif environment == LOCAL or environment == DOCKER:
            if formalization_ml_problem[g.select_ml] == MIP or formalization_ml_problem[g.select_ml] == LP:
                opt_x, opt_value = self.problem.modlize_gurobi_problem(fs, new_s, prev_x=init_x)
            elif formalization_ml_problem[g.select_ml] == NLP:
                opt_x, opt_value = self.problem.modlize_casadi_problem(fs, new_s, prev_x=init_x)
        x, obj, true_obj = self.evaluate_result(self.problem, fs, opt_x, new_s)
        ave_rho, ave_diameter, ave_rdm = -1, -1, -1
        if round(obj, 2) != round(opt_value, 2):
            print("最適解が異なります")
        return x, obj, true_obj, ave_rho, ave_diameter, ave_rdm