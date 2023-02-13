from utility.module import *
from utility.setting import *


def add_polynomial_regression_constraint(model, f, s, x, c, i=0):
    sub = {}
    for idx in range(len(f.coefs) - g.n_user_available_x):
        sub[idx] = model.addVar(vtype=GRB.CONTINUOUS, name=f"sub{idx}", lb=-np.inf)
    model.update()
    # 2次
    f1_i, f2_i = 0, 0
    for idx in range(len(f.coefs) - g.n_user_available_x):
        model.addConstr(f.coefs[idx + g.n_user_available_x] * x[f1_i] * x[f2_i] == sub[idx])
        f2_i += 1
        if f2_i == g.n_user_available_x:
            f1_i += 1
            f2_i = f1_i
    # 1次+2次+ic
    model.addConstr(quicksum(f.coefs[j] * x[j] for j in range(g.n_user_available_x)) + quicksum(sub[idx] * x[idx] for idx in range(len(f.coefs) - g.n_user_available_x)) + f.ic == c, name = f"f{i}")
    return model


def add_polynomial_regression_constraint_casadi(model, x, c, f, s):
    if f.name == POLYNOMIALREGRESSION:
        model.subject_to(casadi_polynomial(x, c, f, s))
    return model       
    

def casadi_polynomial(x, c, f, s):
    base = 0
    # 1次
    for j in range(g.n_user_available_x):
        base += f.coefs[j] * x[j]
    # 2次
    f1_i, f2_i = 0, 0
    for idx in range(len(f.coefs) - g.n_user_available_x):
        base += f.coefs[idx + g.n_user_available_x] * x[f1_i] * x[f2_i]
        f2_i += 1
        if f2_i == g.n_user_available_x:
            f1_i += 1
            f2_i = f1_i
    ans = base + f.ic
    return ans == c