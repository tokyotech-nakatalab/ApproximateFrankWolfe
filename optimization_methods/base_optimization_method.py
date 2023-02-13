from problem.csum_box_feature_constr import CSumBoxFeatureConstrProblem
from problem.csum_disk import CSumDiskConstrProblem
from problem.csum_inequality_constr import CSumInequalityConstrProblem
from problem.advertising_problem import AdvertisingProblem
from utility.setting import *
from utility.tool import *
from problem.csum_nonconstr import *

class BaseOptimizationMethod():
    def __init__(self) -> None:
        self.iteration = 0
        self.max_iteration = n_max_iteration
        self.best_x = None
        self.best_obj = -np.inf

    def set_problem(self):
        if g.select_problem == NONCONSTRAINT:
            self.problem = CSumNonConstraintProblem()
        elif g.select_problem == BOXFEATURECONSTRAINT:
            self.problem = CSumBoxFeatureConstrProblem()
        elif g.select_problem == DISKCONSTRAINT:
            self.problem = CSumDiskConstrProblem()
        elif g.select_problem == INEQUALITYCONSTRAINT:
            self.problem = CSumInequalityConstrProblem()
        elif g.select_problem == REALCONSTRAINT:
            self.problem = AdvertisingProblem()
        self.problem.def_minmax()

    def evaluate_result(self, problem, fs, x, s, print_flg=True):
        xs = np.concatenate([x, s], axis=1)
        obj = problem.objective_f(fs, xs)
        true_obj = problem.objective_true(xs)
        penalty_list = self.problem.penalty_constraint(x)
        if np.sum(penalty_list) != 0:
            obj = obj + self.penalty_weight @ penalty_list
    
        if print_flg:
            if np.sum(penalty_list) == 0:
                print(f"It：{self.iteration}    暫定解：{x}   目的関数値：{obj}  真の目的関数値：{true_obj}")
            else:
                print(f"It：{self.iteration}    暫定解：{x}   目的関数値：{obj}  真の目的関数値：***{true_obj}***")
        return x, obj, true_obj

    def renew_best(self, x, obj, true_obj):
        penalty_list = self.problem.penalty_constraint(x)
        if np.sum(penalty_list) == 0:
            if self.best_obj < obj:
                self.best_obj = obj
                self.best_x = x
                self.best_true_obj = true_obj

    def visualize(self, fs, iter, user_x, history_x, history_y, history_true_y, direct_x=None):
        if g.n_feature == 2 and visualize_optimization_process_status != DONTSEE:
            y = fs[0].predict([[user_x[0, 0], user_x[0, 1]]])[0]
            if direct_x is not None:
                direct_y = fs[0].predict([[direct_x[0, 0], direct_x[0, 1]]])[0]
            true_y = self.problem.calc_c(np.array([user_x[0, 0], user_x[0, 1]]))
            if direct_x is not None:
                direct_true_y = self.problem.calc_c(np.array([direct_x[0, 0], direct_x[0, 1]]))

            if visualize_optimization_process_status == SEEALL or self.iteration == self.max_iteration:
                if direct_x is None:
                    # check_optimization_prosess(fs[0], self.problem, iter, user_x, y, np.array(history_x), history_y)
                    check_optimization_prosess(self.problem.calc_c, self.problem, iter, user_x, y, np.array(history_x), history_true_y)
                else:
                    direct_x = adjust_cordinate_for_process(direct_x)
                    # check_optimization_prosess(fs[0], self.problem, iter, user_x, y, np.array(history_x), history_y, direct_x, direct_y)
                    check_optimization_prosess(self.problem.calc_c, self.problem, iter, user_x, y, np.array(history_x), history_true_y, direct_x, direct_true_y)
            if visualize_optimization_process_status == SEEALL and self.iteration == self.max_iteration:
                create_gif()
            history_x.append(user_x)
            history_y.append(y)
            history_true_y.append(true_y)

    def judge_finish(self, x, prev_x):
        return self.iteration > self.max_iteration # or (prev_x is not None and np.sqrt(np.sum((x - prev_x)**2)) < finish_epsilon)


# def good_fs(fs):
#     return [fs[i].good_obj_mdl for i in range(g.n_item)]