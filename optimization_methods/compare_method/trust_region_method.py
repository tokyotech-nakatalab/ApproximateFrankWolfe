from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from machine_learning.model.approximate_model import *
from optimization_methods.base_optimization_method import *

class TrustRegionMethod(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        self.n_total_user_x = g.n_user_available_x * g.n_item

        #step1 initialize
        delta_x = 0.5 #適当な初期値
        user_x = init_x
        c1, c2, c3, c4 = 0.25, 0.75, 0.5, 2

        self.iteration = 0
        while True:
            #step2 finish judgement
            if self.judge_finish(delta_x):
                break

            #step3 approximate model
            #calc local linear regression ⇒m_k
            mk = [None] * g.n_item
            for i in range(g.n_item):
                mk[i] = MyApproximateModel(fs[i], user_x[i], new_s[i])
                mk[i].set_parameter_for_opt(new_s[i])

            #step4 solve partianl problem
            #solve partial problem ⇒d_k*
            opt_x, opt_value = self.problem.modlize_gurobi_problem(mk, new_s, delta_x, user_x)

            #step5 evaluate model approximation
            xs = np.concatenate([user_x, new_s], axis=1)
            new_xs = np.concatenate([opt_x, new_s], axis=1)

            delta_fk = self.problem.objective_f(fs, new_xs) - self.problem.objective_f(fs, xs)
            if delta_fk == 0.0:
                delta_fk = 0.01
            delta_mk = self.problem.objective_f(mk, new_xs) - self.problem.objective_f(mk, xs)
            if delta_mk == 0.0:
                delta_mk = 0.01
            rho = delta_fk / delta_mk

            #step6 renew x
            if rho >= c1: #c1未満なら何もしない
                user_x = opt_x

            #step7 renew delta
            if  rho >= c2: #
                delta_x = c4 * delta_x
            elif c1 <= rho and rho <= c2: #
                delta_x = delta_x
            else:
                delta_x = c3 * delta_x

            x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s)
            self.renew_best(x, obj, true_obj)

            self.iteration += 1
        return self.best_x, self.best_obj, self.best_true_obj
            

    def judge_finish(self, delta_x):
        return self.iteration > self.max_iteration # or delta_x < 0.001
