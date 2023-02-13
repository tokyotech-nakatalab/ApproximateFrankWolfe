from utility.module import *


def add_randomforest_regression_constraint(model, f, x, y, index=0):
    n_threshold = len(f.threshold)
    n_delta = len(f.pred)
    l = model.addMVar(shape=n_threshold, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"l{index}")
    d = model.addMVar(shape=n_delta, vtype=GRB.BINARY, name=f"d{index}")
    model.update()

    model.addConstr(l @ f.threshold == x)
    model.addConstr(d @ f.pred == y)
    model.addConstr(l.sum() == 1)
    for i in range(n_threshold):
        if i == 0:
            model.addConstr(l[i] - d[i] <= 0)
        elif i == n_threshold - 1:
            model.addConstr(l[i] - d[i-1] <= 0)
        else:
            model.addConstr(l[i] - d[i] - d[i-1] <= 0)
    model.addConstr(d.sum() == 1)
    return model