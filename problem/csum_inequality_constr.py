from problem.base_problem import *
from utility.tool import tic2, toc2

class CSumInequalityConstrProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        if g.n_feature == 10:
            self.M = 45
        elif g.n_feature == 30:
            self.M = 100
        self.constr = [5, 2, 3, 7, 10, 4, 7, 3, 1, 5]
        # self.constr = [7, 14, 11, 7, 9, 13, 7, 6, 8, 9]
        # self.constr = [7, 14, 11, 7, 9, 13, 7, 15, 8, 9]
        self.constr = [12] * g.n_feature
        self.constr = [8] * 10
        self.min_constr = [[min(self.constr[i], self.constr[j]) for i in range(len(self.constr))] for j in range(len(self.constr))]
        
        if g.n_feature == 10:
            self.constr_idx = [[0, 2], [4, 6], [7, 8],
                            #    [5, 2], [9, 1], [3, 4],
                            #    [5, 1], [0, 3], [4, 9], [2, 8]
                               ]
        elif g.n_feature == 30:
            self.constr_idx = [[0, 2], [4, 10], [15, 22], [17, 8], [9, 11], [13, 4], [5, 1], [0, 3], [4, 9], [2, 8], [27, 21],
                               [5, 29], [12, 28], [3, 24], [6, 19], [7, 26], [8, 21], [14, 16], [16, 20], [18, 19], [23, 6], [25, 12]]

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

            # 問題に関わる制約
            for i in range(g.n_item):
                self.model.addConstr(quicksum(self.X[i][j] for j in range(g.n_feature)) <= self.M)
                for cons in self.constr_idx:
                    j, k = cons[0], cons[1]
                    self.model.addConstr(self.X[i][j] + self.X[i][k] <= self.min_constr[j][k])
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

        # 問題に関わる制約
        for i in range(g.n_item):
            # model.subject_to(X[i][0] + X[i][1] == self.M)
            model.subject_to(sum(X[i][j] for j in range(g.n_feature)) <= self.M)
            for cons in self.constr_idx:
                j, k = cons[0], cons[1]
                model.subject_to(X[i][j] + X[i][k] <= self.min_constr[j][k])

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
            

        # 問題に関わる制約
        for i in range(g.n_item):
            model += pulp.lpSum(X[i][j] for j in range(g.n_user_available_x)) <= self.M
            for cons in self.constr_idx:
                j, k = cons[0], cons[1]
                model += X[i][j] + X[i][k] <= self.min_constr[j][k]   

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

    def penalty_capcity(self, x):
        val = np.sum(x) - self.M
        p = max(val, 0) ** 2
        return [p]

    # def penalty_constraint(self, x):
    #     return np.array(self.penalty_capcity(x))

    def penalty_xmax(self, x):
        p = 0.
        for cons in self.constr_idx:
            j, k = cons[0], cons[1]
            try:
                val = x[0][j] + x[0][k] - self.min_constr[j][k]
            except:
                val = x[j] + x[k] - self.min_constr[j][k]
            p += (max(val, 0)) ** 2
        return [p]

    # def penalty_x1min(self, x):
    #     val = self.x1_min_bound - np.sum(x, axis=0)[1]
    #     p = (np.max(val, 0)) ** 2
    #     return [p]

    def penalty_constraint(self, x):
        penalty = np.array(self.penalty_capcity(x) + self.penalty_xmax(x))
        penalty[penalty < 10 ** -4] = 0.
        return penalty