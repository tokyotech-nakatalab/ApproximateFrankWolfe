from select import select
from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from machine_learning.model.approximate_model import *
from solvers.base_optimization_method import *

class FrankWolfe2(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        self.n_total_user_x = g.n_user_available_x * g.n_item

        #step1 initialize
        user_x = init_x
        prev_x = None

        min_rho_list, ave_diameter_list, rho_diam_near_list = [], [], []
        g.search_time, g.fit_time, g.svd_time, g.culcd_time, g.modelize_time, g.solve_time = 0, 0, 0, 0, 0, 0
        self.finish_cnt = 0
        alpha = 0.0
        self.iteration = 0
        history_x, history_y, history_true_y = [], [], []
        self.nearest_history = {}
        mk = [None] * g.n_item
        while True:
            #step2 finish judgement
            if self.judge_finish(user_x, prev_x):
                break
            # print(f"iteration:{self.iteration}")

            #step3 local linear model
            #calc local linear regression ⇒m_k
            for i in range(g.n_item):
                xs = np.concatenate([user_x[i], new_s[i]], axis=0)
                (distances, min_rho, diameter), nearest_idx, is_seen = fs[i].fit_xk(xs)
                min_rho_list.append(1 / min_rho)
                ave_diameter_list.append(diameter)
                rho_diam_near_list.append(diameter**2 / min_rho * np.sqrt(g.n_nearest))
                if not is_seen:
                    mk[i] = fs[i]
                    mk[i].set_parameter()
                    mk[i].set_parameter_for_opt(new_s[i])
                else:
                    break

            if self.iteration != 0:
                prev_x = np.copy(user_x)
                if self.iteration != 1 and is_seen:
                    opt_x, opt_value = self.nearest_history[tuple(nearest_idx)], np.nan 
                elif environment == LOCAL or environment == DOCKER:
                    opt_x, opt_value = self.problem.modlize_gurobi_problem(mk, new_s, prev_x=user_x)
                    self.nearest_history[tuple(nearest_idx)] = opt_x
                elif environment == TSUBAME:
                    opt_x, opt_value = self.problem.modlize_pulp_problem(mk, new_s, prev_x=user_x)
                    self.nearest_history[tuple(nearest_idx)] = opt_x
                xs = np.concatenate([user_x, new_s], axis=1)

                alpha = 2 / (self.iteration + 2 - 1) # 初回に動かない分の補正
                # alpha = 1 / (self.iteration + 1000 - 1)
                user_x = user_x + alpha * (opt_x - user_x)
            else:
                # print("初回イテレーションでは移動しません")
                opt_x = None
            x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s, print_flg=False)

            # x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s)
            self.renew_best(x, obj, true_obj)
            self.iteration += 1
        ave_rho = sum(min_rho_list) / len(min_rho_list)
        ave_diameter = sum(ave_diameter_list) / len(ave_diameter_list)
        # ave_rdnear = sum(rho_diam_near_list) / len(rho_diam_near_list)
        ave_rdnear = sum(rho_diam_near_list) / (len(rho_diam_near_list) + 1)
        # ave_rdnear = max(rho_diam_near_list)
        if g.select_ml == WEIGHTEDLINEARREGRESSION:
            return self.best_x, self.best_obj, self.best_true_obj
        elif g.select_ml == ANNLINEARREGRESSION or g.select_ml == KNNLINEARREGRESSION:
            return x, obj, true_obj, ave_rho, ave_diameter, ave_rdnear

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

    def judge_finish(self, x, prev_x):
        if prev_x is None:
            return False
        dif_sum = np.sum(np.abs(x - prev_x))
        # print(dif_sum)
        if dif_sum < 0.1:
            self.finish_cnt += 1
        else:
            self.finish_cnt = 0
        return self.finish_cnt >= 10 or self.iteration > self.max_iteration 