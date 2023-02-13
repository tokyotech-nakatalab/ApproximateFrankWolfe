from utility.module import *


def add_quadratic_regression_constraint(model, f, s, x, c, index=0):
    model.update()
    model.addConstr(f.coefs @ x + x @ f.B @ x == c, name = f"f{index}")
    return model