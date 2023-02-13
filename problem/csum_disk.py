from problem.base_problem import *

class CSumDiskConstrProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()


    def modlize_gurobi_problem(self, fs, s, delta_x=None, prev_x=None):
        """
        入力:回帰した関数fの集合
        x:1次元, y:1次元の重回帰.が二種類存在.xの総和が10以内で二つの回帰の総和の最大値
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
            model.addConstr(X[i][0]**2 + X[i][1]**2 <= 2)

        #信頼領域を設定する場合
        if not delta_x is None:
            x2 = delta_x ** 2
            model.addConstr(quicksum((X[i][j] - prev_x[i][j]) * (X[i][j] - prev_x[i][j]) for j in range(g.n_user_available_x) for i in range(g.n_item)) <= x2)

        return self.do_optimize(model, X, C)


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
                X[i][j] = model.variable()
                model.subject_to(X[i][j] >= self.min_bounds[i][j])
                model.subject_to(X[i][j] <= self.max_bounds[i][j])
                model.set_initial(X[i][j], prev_x[i][j])

        # 機械学習に関わる制約
        model.ml = {}
        for i in range(g.n_item):
            model = self.add_constraint_forecast_casadi(model, fs[i], s[i], X[i], C[i])

        # 問題に関わる制約
        for i in range(g.n_item):
            model.subject_to(self.casadi_disc(X[i]))

        # 目的関数を設定
        obj = self.casadi_objective(C)
        model.minimize(obj)

        x_list, opt_obj = self.do_optimize_casadi(model, X, C, obj)
        return x_list, -opt_obj


    def objective(self, c):
        return np.sum(c)


    def casadi_objective(self, C):
        obj = -1 * sum(C.values())
        return obj


    def casadi_disc(self, x):
        return np.sum(x**2) <= 2


    def penalty_disk(self, x):
        p_list = []
        for i in range(g.n_item):
            p_list.append((max([np.sum(x[i]**2) - 2, 0])) ** 2)
        return [sum(p_list)]

    def penalty_constraint(self, x):
        return np.array(self.penalty_disk(x))