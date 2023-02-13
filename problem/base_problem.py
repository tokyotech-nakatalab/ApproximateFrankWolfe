from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from solvers.ml_modelize.linear_regression import add_linear_regression_constraint, add_linear_regression_constraint_pulp
from solvers.ml_modelize.random_forest import add_randomforest_regression_constraint
from solvers.ml_modelize.supportvector_regression import add_supportvector_regression_constraint, add_supportvector_regression_constraint_casadi
from solvers.ml_modelize.cga2m_regression import add_cga2m_regression_constraint
from solvers.ml_modelize.polynomial_regression import add_polynomial_regression_constraint, add_polynomial_regression_constraint_casadi

class BaseProblem(BaseOfData):
    def __init__(self) -> None:
        super().__init__()

    def def_minmax(self):
        self.min_bounds = self.data_generator.min_bounds
        self.max_bounds = self.data_generator.max_bounds
        try:
            self.real_max_bounds = self.data_generator.original_max_bounds
            self.real_min_bounds = self.data_generator.original_min_bounds
        except:
            self.real_max_bounds = self.data_generator.max_bounds
            self.real_min_bounds = self.data_generator.min_bounds       

    def do_optimize(self, model, X, C, verbose=False):
        # 解を求める計算
        if verbose:
            print("↓点線の間に、Gurobi Optimizerからログが出力")
            print("-" * 40)
            model.optimize()
            print("-" * 40)
            print()
        else:
            model.Params.outputFlag = 0
            model.optimize()
        x_opt_list, c_opt_list = [], []
        if model.Status == GRB.OPTIMAL:
            for i in range(g.n_item):
                for j in range(g.n_user_available_x):
                    x_opt_list += [X[i][j].X]
                c_opt_list += [C[i].X]
            x_opt_list = np.array(x_opt_list).reshape(g.n_item, g.n_user_available_x)
            val_opt = model.ObjVal
            # print(f"最適解は x = {x_opt_list}, c = {c_opt_list}")
            # print(f"最適値は {val_opt}")
            return x_opt_list, val_opt
        elif model.Status == GRB.SUBOPTIMAL:
            for i in range(g.n_item):
                for j in range(g.n_user_available_x):
                    x_opt_list += [X[i][j].X]
                c_opt_list += [C[i].X]
            x_opt_list = np.array(x_opt_list).reshape(g.n_item, g.n_user_available_x)
            val_opt = model.ObjVal
            # print(f"準最適解は x = {x_opt_list}, c = {c_opt_list}")
            # print(f"準最適値は {val_opt}")
            return x_opt_list, val_opt
        else:
            print(model.Status)
            print("最適解が求まりませんでした")
            return -1


    def do_optimize_casadi(self, model, X, C, obj):
        p_opts = {'print_time': False}
        s_opts = {'print_level' : 0,
                  'tol' : 1E-3,
                #   "max_cpu_time": 1.0, 
                #   'max_iter' : 10
                }
        model.solver('ipopt', p_opts, s_opts)
        # model.solver('mumps', p_opts, s_opts)
        sol = model.solve() # 最適化計算を実行

        x_opt_list, c_opt_list = [], []
        for i in range(g.n_item):
            for j in range(g.n_user_available_x):
                x_opt_list += [sol.value(X[i][j])]
            c_opt_list += [sol.value(C[i])]
        x_opt_list = np.array(x_opt_list).reshape(g.n_item, g.n_user_available_x)
        val_opt = sol.value(obj)
        return x_opt_list, val_opt


    def do_optimize_pulp(self, model, X, C, obj):
        sol = model.solve(pulp.PULP_CBC_CMD(msg = False))

        x_opt_list, c_opt_list = [], []
        for i in range(g.n_item):
            for j in range(g.n_user_available_x):
                x_opt_list += [X[i][j].value()]
            c_opt_list += [C[i].value()]
        x_opt_list = np.array(x_opt_list).reshape(g.n_item, g.n_user_available_x)
        val_opt = model.objective.value()
        return x_opt_list, val_opt

    def objective_f(self, fs, xs):
        c = np.zeros(g.n_item)
        for i in range(g.n_item):
            if g.select_ml != WEIGHTEDLINEARREGRESSION:
                try:
                    c[i] = fs[i].predict(xs[i])
                except:
                    c[i] = fs[i].predict([xs[i]])
                check_correct_predict(fs[i], xs[i], c[i])
            else:
                try:
                    c[i] = fs[i].good_obj_mdl.predict(xs[i])
                except:
                    c[i] = fs[i].good_obj_mdl.predict([xs[i]])
        return self.objective(c)

    def objective_true(self, xs):
        c = np.zeros(g.n_item)
        for i in range(g.n_item):
            c[i] = self.calc_c(xs[i])
        return self.objective(c)

    def penalty_constraint(self, x):
        return np.array([0.])

    def add_constraint_forecast(self, model, f, s, x, c, index):
        if f.name == LINEARREGRESSION: #Gurobi
            return add_linear_regression_constraint(model, f, s, x, c, index)
        elif f.name == RANDOMFOREST: #Gurobi
            return add_randomforest_regression_constraint(model, f, s, x, c, index)
        elif f.name == SVRLINEAR: #Gurobi
            return add_supportvector_regression_constraint(model, f, s, x, c, index)    
        elif f.name == CGA2M: #Gurobi
            return add_cga2m_regression_constraint(model, f, x, c, s, self.min_bounds, index)
        elif f.name == POLYNOMIALREGRESSION: # Gurobi
            return add_polynomial_regression_constraint(model, f, s, x, c)
        # elif f.name == LIGHTGBM: #Gurobi
        #     return add_lightgbm_regression_constraint(model, f, s, x, c, index)

    def add_constraint_forecast_casadi(self, model, f, s, x, c):
        if f.name == SVRPOLY or f.name == SVRGAUSS: #casadi(IPOPT)
            return add_supportvector_regression_constraint_casadi(model, x, c, f, s)
        elif f.name == POLYNOMIALREGRESSION: # casadi(IPOPT)
            return add_polynomial_regression_constraint_casadi(model, x, c, f, s)

    def add_constraint_forecast_pulp(self, model, f, s, x, c):
        if f.name == LINEARREGRESSION: #pulp(CBC)
            return add_linear_regression_constraint_pulp(model, x, c, f, s)

    def check_penalty(self, x):
        penalty = self.penalty_constraint(x)
        if np.sum(penalty) != 0:
            return False
        for i in range(g.n_item):
            for j in g.user_available_x:
                if self.min_bounds[i][j] > x[j]:
                    return False
                if self.max_bounds[i][j] < x[j]:
                    return False
        return True


    def casadi_transregion(model):
        sum_ = 0
        for i in range(g.n_item):
            for j in g.user_available_x:
                sum_ += (model.x[i][j] - g.prev_x[i][j]) ** 2
        return sum_ <= g.delta_x ** 2


def check_correct_predict(f, xs, c):
    if g.select_ml == SVRPOLY or g.select_ml == SVRGAUSS:
        c_m = f.predict_manual(xs)
        if round(c, 3) != round(c_m, 3):
            print("手動の予測器との結果が異なります")