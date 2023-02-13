from utility.module import *
from utility.setting import *


def add_linear_regression_constraint(model, f, s, x, c, i=0):
    model.update()
    model.addConstr(quicksum(f.coefs[j] * x[j] for j in range(g.n_user_available_x)) + f.ic == c, name = f"f{i}")
    return model


def add_linear_regression_constraint_casadi(model, x, c, f, s):
    model.subject_to(casadi_linear_regression(x, c, f))
    return model


def casadi_linear_regression(x, c, f):
    ans = np.sum(np.array([f.coefs[i] * x[i] for i in range(g.n_user_available_x)])) + f.ic
    return ans == c


def add_linear_regression_constraint_pulp(model, x, c, f, s):
    model += pulp.lpSum(f.coefs[i] * x[i] for i in range(g.n_user_available_x)) + f.ic == c
    return model