from utility.module import *

M = 10000

def add_step_combination_constraint(model, f, x, y, index=0):
    s = []
    for i in range(x.shape[0]):
        s.append(model.addVar(vtype=GRB.BINARY, name=f"s{(index,i)}"))
        model.addConstr(M * s[i] > x - f.thresholds[i])
        model.addConstr(-M * s[i] <= x - f.thresholds[i])
        model.addConstr(f.coefs[i] * s[i] == y, name = f"f{index}")
    return model