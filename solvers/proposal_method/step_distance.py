from select import select
from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from machine_learning.model.approximate_model import *
from solvers.base_optimization_method import *

class StepDistance(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        self.n_total_user_x = g.n_user_available_x * g.n_item

        #step1 initialize
        user_x = init_x

        alpha = 0.0
        self.iteration = 0
        history_x, history_y, history_true_y = [], [], []
        while True:
            #step2 finish judgement
            if self.judge_finish():
                break

            #step3 local linear model
            #calc local linear regression ⇒m_k
            mk = [None] * g.n_item
            for i in range(g.n_item):
                xs = np.concatenate([user_x[i], new_s[i]], axis=0)
                distances = fs[i].fit_xk(xs)
                mk[i] = fs[i]
                mk[i].set_parameter()
                mk[i].set_parameter_for_opt(new_s[i])

            if self.iteration != 0:
                if environment == LOCAL or environment == DOCKER:
                    opt_x, opt_value = self.problem.modlize_gurobi_problem(mk, new_s)
                elif environment == TSUBAME:
                    opt_x, opt_value = self.problem.modlize_casadi_problem(fs, new_s)
                xs = np.concatenate([user_x, new_s], axis=1)

                # alpha = 2 / (self.iteration + 2)
                # dif_vec = (opt_x - user_x)
                # dif_vec_dist = np.linalg.norm(dif_vec, 2)
                # th_dist = distances[distances.size // 2]
                # if dif_vec_dist * alpha > th_dist:
                #     user_x = user_x + dif_vec / dif_vec_dist * th_dist
                # else:
                #     user_x = user_x + alpha * dif_vec
                alpha = 2 / (self.iteration + 2)
                user_x = user_x + alpha * (opt_x - user_x)
            else:
                print("初回イテレーションでは移動しません")
                opt_x = None
            self.visualize(mk, self.iteration, user_x, history_x, history_y, history_true_y, opt_x)

            x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s)
            self.renew_best(x, obj, true_obj)
            self.iteration += 1
        if g.select_ml == WEIGHTEDLINEARREGRESSION:
            return self.best_x, self.best_obj, self.best_true_obj
        elif g.select_ml == ANNLINEARREGRESSION:
            return x, obj, true_obj

    def check_regression_shape(self, f, problem, s, x, y):
        n_density = 1000
        feature0_x = (np.linspace(problem.min_bounds[0][0], problem.max_bounds[0][0], n_density)).reshape(n_density, -1)
        testest_x = feature0_x.copy()
        if g.n_user_available_x != 1:
            zero_x = np.zeros((n_density, g.n_user_available_x-1))
            testest_x = np.concatenate([feature0_x, zero_x], axis=1)
        if g.n_environment_s != 0:
            test_s = np.array([list(s)[0] for _ in range(n_density)])
            testest_x = np.concatenate([testest_x, test_s], axis=1)

        fig = plt.figure()
        plt.plot(feature0_x, f.predict(testest_x))
        plt.scatter(x, y)
        plt.show()
        plt.close()