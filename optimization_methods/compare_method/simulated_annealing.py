from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from optimization_methods.base_optimization_method import *

class SimulatedAnnealing(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        user_x = init_x
        self.eta = 0.1
        self.penalty_weight = np.array([2.] * len(self.problem.penalty_constraint(user_x)))
        prev_fitness = self.evaluate(fs, user_x, new_s)

        N = self.max_iteration
        T = 10.0

        history_x, history_y, history_true_y = [], [], []
        self.iteration = 0
        while True:
            if self.judge_finish(user_x, None):
                break

            t = T - T*(float(self.iteration)/N)# 温度更新
            if self.iteration != 0:
                new_user_x = self.renew_user_x(user_x) # 探索
                fitness = self.evaluate(fs, new_user_x, new_s)# 評価関数計算
                # 更新判定
                if fitness < prev_fitness:
                    user_x = copy.deepcopy(new_user_x)
                else:
                    p = np.exp(-(fitness - prev_fitness) / t) # 更新確率計算
                    print(f"確率{p}")
                    # p = 0.
                    if np.random.rand() <= p:
                        user_x = copy.deepcopy(new_user_x)
                prev_fitness = self.evaluate(fs, user_x, new_s)
                self.update_penalty_weight(user_x)
            else:
                print("初回イテレーションでは移動しません")
                opt_x = None
            self.visualize(fs, self.iteration, user_x, history_x, history_y, history_true_y, opt_x)
            x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s)

            self.renew_best(x, obj, true_obj)
            self.iteration += 1
        ave_rho, ave_diameter, ave_rdm = -1, -1, -1
        penalty_list = self.problem.penalty_constraint(self.best_x)
        if np.sum(penalty_list) == 0:
            return self.best_x, self.best_obj, self.best_true_obj, ave_rho, ave_diameter, ave_rdm
        else:
            dummy_true_obj = 0.1
            return self.best_x, self.best_obj, dummy_true_obj, ave_rho, ave_diameter, ave_rdm

    def renew_user_x(self, user_x):
        # d_x =  (self.problem.max_bounds[0][0] - self.problem.min_bounds[0][0]) * np.random.rand(g.n_item, g.n_user_available_x) + self.problem.min_bounds[0][0]
        d_x = np.random.rand(g.n_item, g.n_user_available_x) - 0.5
        new_user_x = user_x + d_x
        for i in range(g.n_item):
            for j in g.user_available_x:
                if new_user_x[i][j] < self.problem.min_bounds[i][j]:
                    new_user_x[i][j] = self.problem.min_bounds[i][j]
                elif new_user_x[i][j] > self.problem.max_bounds[i][j]:
                    new_user_x[i][j] = self.problem.max_bounds[i][j]
        return new_user_x

    def update_penalty_weight(self, user_x):
        penalty_list = self.problem.penalty_constraint(user_x)
        penalty_renew_flg = False
        for i in range(penalty_list.size):
            if penalty_list[i] != 0:
                self.penalty_weight[i] = self.penalty_weight[i] * (1 + self.eta * self.penalty_weight[i] / sum(self.penalty_weight))
                penalty_renew_flg = True
        if not penalty_renew_flg:
            for i in range(self.penalty_weight.size):
                self.penalty_weight[i] = self.penalty_weight[i] * (1 - self.eta)

    def evaluate(self, fs, x, s):
        c_hats = []
        #まずは予測
        for i in range(g.n_item):
            x_i = x[i]
            s_i = s[i]
            if x_i.size == 0:
                xs = s_i
            elif s_i.size == 0:
                xs = x_i
            else:
                xs = np.concatenate([x_i, s_i])

            c_hat = fs[i].predict([xs])
            c_hats.append(c_hat)

        #予測したc^を使って目的関数値を求める
        penalty_list = self.problem.penalty_constraint(x)
        obj = self.problem.objective(c_hats) - self.penalty_weight @ penalty_list

        #最小化にするためにマイナスをかけてreturn
        return -obj