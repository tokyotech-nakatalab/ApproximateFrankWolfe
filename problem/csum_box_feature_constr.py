from problem.base_problem import *

class CSumBoxFeatureConstrProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.M = 1

    def modlize_gurobi_problem(self, fs, s, delta_x=None, prev_x=None):
        """
        入力：回帰した関数fの集合
        x:1次元, y:1次元の重回帰．が二種類存在．xの総和が10以内で二つの回帰の総和の最大値
        """

        X = {}
        C = {}

        model = Model(name = "taxi")
        # 変数の生成
        for i in range(g.n_item):
            X[i] = {}
            for j in range(g.n_user_available_x):
                X[i][j] = model.addVar(vtype=GRB.CONTINUOUS, lb=self.min_bounds[i][j], ub=self.max_bounds[i][j], name=f"x{i,j}")
            C[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"c{i}", lb=-np.inf)


        # 機械学習に関わる制約
        for i in range(g.n_item):
            model = self.add_constraint_forecast(model, fs[i], s[i], X[i], C[i], index=i)


        # 目的関数を設定
        model.setObjective(quicksum(C[i] for i in range(g.n_item)), sense = GRB.MAXIMIZE)

        # 問題に関わる制約
        for i in range(g.n_item):
            model.addConstr(X[i][0] + X[i][1] == self.M)
            model.addConstr(X[i][2] + X[i][4] == self.M)
            model.addConstr(X[i][0] + X[i][3] == self.M)

        return self.do_optimize(model, X, C, g.n_item)


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
            model.subject_to(X[i][0] + X[i][1] == self.M)
            model.subject_to(X[i][2] + X[i][4] == self.M)
            model.subject_to(X[i][0] + X[i][3] == self.M)

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
            model += pulp.lpSum(X[i][j] for j in range(g.n_user_available_x)) == self.M
            model += X[i][0] + X[i][1] == self.M
            model += X[i][2] + X[i][4] == self.M
            model += X[i][0] + X[i][3] == self.M

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
        p = (np.max(val, 0)) ** 2
        return [p]

    def penalty_constraint(self, x):
        return np.array(self.penalty_capcity(x))

    # def penalty_x0max(self, x):
    #     val = np.sum(x, axis=0)[0] - self.x0_max_bound
    #     p = (np.max(val, 0)) ** 2
    #     return [p]

    # def penalty_x1min(self, x):
    #     val = self.x1_min_bound - np.sum(x, axis=0)[1]
    #     p = (np.max(val, 0)) ** 2
    #     return [p]

    # def penalty_constraint(self, x):
    #     return np.array(self.penalty_capcity(x) + self.penalty_x0max(x) + self.penalty_x1min(x))