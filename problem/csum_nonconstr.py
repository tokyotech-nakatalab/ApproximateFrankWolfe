from problem.base_problem import *
import utility.gloabl_values as g
from utility.tool import tic2, toc2


class CSumNonConstraintProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()

    def modlize_gurobi_problem(self, fs, s, delta_x=None, prev_x=None):
        """
        入力：回帰した関数fの集合
        x:1次元, y:1次元の重回帰．が二種類存在．xの総和が10以内で二つの回帰の総和の最大値
        """

        tic2()
        if self.first_do:
            self.X = {}
            self.C = {}
            self.ml_constr = {}

            self.model = Model(name = "taxi")
            # 変数の生成
            for i in range(g.n_item):
                self.X[i] = {}
                for j in range(g.n_user_available_x):
                    self.X[i][j] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=self.min_bounds[i][j], ub=self.max_bounds[i][j], name=f"x{i,j}")
                self.C[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"c{i}", lb=-np.inf)
            self.model.update()


            # 目的関数を設定
            self.model.setObjective(quicksum(self.C[i] for i in range(g.n_item)), sense = GRB.MAXIMIZE)
            self.first_do = False
        else:
            for i in range(g.n_item):
                self.model.remove(self.ml_constr[i])

        # 機械学習に関わる制約
        for i in range(g.n_item):
            self.model, self.ml_constr[i] = self.add_constraint_forecast(self.model, fs[i], s[i], self.X[i], self.C[i], index=i)

        g.modelize_time += toc2()
        tic2()
        x_opt_list, val_opt = self.do_optimize(self.model, self.X, self.C)
        g.solve_time += toc2()
        return x_opt_list, val_opt
        # return self.do_optimize(model, X, C)


    def modlize_casadi_problem(self, fs, s, prev_x=None):
        model = casadi.Opti()
        X, C = {}, {}
        for i in range(g.n_item):
            X[i] = {}
            for j in range(g.n_user_available_x):
                X[i][j] = model.variable()
            C[i] = model.variable()

        # 変数に関わる制約
        for i in range(g.n_item):
            for j in range(g.n_user_available_x):
                model.subject_to(X[i][j] >= self.min_bounds[i][j])
                model.subject_to(X[i][j] <= self.max_bounds[i][j])
                if prev_x is not None:
                    model.set_initial(X[i][j], prev_x[i][j])

        # 機械学習に関わる制約
        model.ml = {}
        for i in range(g.n_item):
            model = self.add_constraint_forecast_casadi(model, fs[i], s[i], X[i], C[i])

        # 目的関数を設定
        obj = self.casadi_objective(C)
        model.minimize(obj)

        x_list, opt_obj = self.do_optimize_casadi(model, X, C, obj)
        return x_list, -opt_obj


    def modlize_pulp_problem(self, fs, s, prev_x=None):
        if prev_x is not None:
            target_index = prev_x > self.max_bounds
            if np.sum(target_index) != 0:
                prev_x[target_index] = np.array(self.max_bounds)[target_index]
            target_index = prev_x < self.min_bounds
            if np.sum(target_index) != 0:
                prev_x[target_index] = np.array(self.min_bounds)[target_index]
        model = pulp.LpProblem(sense = pulp.LpMaximize)
        X, C = {}, {}
        for i in range(g.n_item):
            X[i] = {}
            for j in range(g.n_user_available_x):
                X[i][j] = pulp.LpVariable(f'x{(i,j)}', lowBound=self.min_bounds[i][j], upBound=self.max_bounds[i][j])
                # 初期解の設定
                if prev_x is not None:
                    X[i][j].setInitialValue(prev_x[i][j])
            C[i] = pulp.LpVariable(f'c{i}')
                    

        # 機械学習に関わる制約
        for i in range(g.n_item):
            model = self.add_constraint_forecast_pulp(model, fs[i], s[i], X[i], C[i])

        # 目的関数を設定
        obj = pulp.lpSum(C[i] for i in range(g.n_item))
        model.setObjective(obj)

        x_list, opt_obj = self.do_optimize_pulp(model, X, C, obj)
        return x_list, opt_obj


    def objective(self, c):
        return np.sum(c)

    def casadi_objective(self, C):
        obj = -1 * sum(C.values())
        return obj