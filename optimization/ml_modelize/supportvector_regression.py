from utility.module import *
from utility.constant import *
import utility.gloabl_values as g

def add_supportvector_regression_constraint(model, f, x, y, index=0):
    if f.name == SVRLINEAR:
        #変数を定義
        sum_xsv = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, name=f"sum{index}")
        model.update()

        #制約定義
        for i in range(f.n_sv):
            model.addConstr(f.sv[i] @ x == sum_xsv[i])
        model.addConstr(quicksum(f.coef[i] * sum_xsv[i] for i in range(f.n_sv)) + f.ic == y)

        additional_var = [sum_xsv]
        return model, additional_var

def add_supportvector_regression_constraint_casadi(model, x, c, f, s):
    if f.name == SVRPOLY:
        model.subject_to(casadi_svrpoly(x, c, f, s))
    elif f.name == SVRGAUSS:
        model.subject_to(casadi_svrgauss(x, c, f, s))
    return model       
    

def casadi_svrpoly(x, c, f, s):
    kernel_value = []
    for i in range(f.n_sv):
        base = 0.
        for j in range(g.n_user_available_x):
            base += f.sv[i][j] * x[j]
        for j in g.environment_s:
            s_index = j - g.n_user_available_x
            base += f.sv[i][j] * s[s_index]
        base_gamma = base * f.gamma
        base_gamma_coef0 = base_gamma + f.coef0
        base_gamma_coef0_degree = base_gamma_coef0 ** f.degree
        kernel_value.append(base_gamma_coef0_degree)
    ans = sum([f.coef[i] * kernel_value[i] for i in range(f.n_sv)]) + f.ic
    return ans == c


def casadi_svrgauss(x, c, f, s):
    kernel_value = []
    for i in range(f.n_sv):
        base = 0.
        for j in range(g.n_user_available_x):
            base += (f.sv[i][j] - x[j]) ** 2
        for j in g.environment_s:
            s_index = j - g.n_user_available_x
            base += (f.sv[i][j] - g.s[s_index]) ** 2
        base_gamma = -f.gamma * base
        exp = np.exp(base_gamma)
        kernel_value.append(exp)
    ans = sum([f.coef[i] * kernel_value[i] for i in range(f.n_sv)]) + f.ic
    return ans == c




    # if f.name == SVRPOLY:  
    #     deg_list = [0] * (f.degree + 1)
    #     deg_list[0] = 1
    #     #変数を定義
    #     xsv = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, name=f"xsv{index}")
    #     xsvg = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, name=f"xsvg{index}")
    #     xsvg_c = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, name=f"xsvg_c{index}")
    #     model.update()

    #     #制約定義
    #     for i in range(f.n_sv):
    #         model.addConstr(f.sv[i] @ x == xsv[i])
    #         model.addConstr(xsv[i] * f.gamma == xsvg[i])
    #         model.addConstr(xsvg[i] + f.coef0 == xsvg_c[i])
    #         model.addGenConstrPoly(xsvg_c[i], y, deg_list)
    #     model.addConstr(quicksum(f.coef[i] * xsvg_c[i] for i in range(f.n_sv)) + f.ic == y)

    #     additional_var = []
    #     return model, additional_var
    # elif f.name == SVRGAUSS:
    #     #変数を定義
    #     x_sv = []
    #     pow_x_sv = []
    #     for i in range(f.n_sv):
    #         pow_x_sv.append([])
    #         x_sv.append([])
    #         for j in range(f.x_dim):
    #             x_sv[i].append(model.addVar(vtype=GRB.CONTINUOUS, name=f"x_sv{index,i,j}"))
    #             pow_x_sv[i].append(model.addVar(vtype=GRB.CONTINUOUS, name=f"pow{index,i,j}"))
    #     #pow_x_sv = model.addMVar(shape=(f.n_sv, f.x_dim), vtype=GRB.CONTINUOUS, name=f"pow{index}")
    #     power = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, name=f"p{index}")
    #     exp = model.addMVar(shape=f.n_sv, vtype=GRB.CONTINUOUS, lb=0, name=f"exp{index}")
    #     model.update()

    #     #制約定義
    #     for i in range(f.n_sv):
    #         for j in range(f.x_dim):
    #             model.addConstr((x[j] - f.sv[i][j]) == x_sv[i][j])
    #             model.addQConstr(x_sv[i][j] * x_sv[i][j] == pow_x_sv[i][j])
    #         # model.addConstr(-f.gamma * quicksum(pow_x_sv[i][j] for j in range(f.x_dim)) == power[i])
    #         # model.addGenConstrExp(power[i], exp[i]) #expを本来の制約や目的関数に使用すればOK
    #     # #model.addConstr(quicksum(power[i] for i in range(f.n_sv)) == y)
    #     # model.addConstr(quicksum(f.coef[i] * exp[i] for i in range(f.n_sv)) + f.ic == y)

    #     model.addQConstr(x[0] * x[0] == y)

    #     model.Params.NonConvex = 2
    #     return model