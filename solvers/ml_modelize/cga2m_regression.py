from utility.module import *
from utility.setting import *

def add_cga2m_regression_constraint(model, f, x, y, s, min_bound, index=0):
    pred_by_environment = 0
    sum_feature_mean = 0

    #ユーザが動かせる変数に絞ってgurobi用変数を定義
    L, D, M, P ={}, {}, {}, {}
    for i in f.main_feature_area.keys():
        if i in g.user_available_x: # 変数がユーザが動かせる変数なら
            n_threshold = len(f.main_feature_area[i]['threshold'])
            n_delta = len(f.main_feature_area[i]['value'])
            l_name, d_name, p_name = f"l{index}: {i}", f"d{index}: {i}", f"p{index}: {i}"
            L[l_name] = model.addMVar(shape=n_threshold, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=l_name)
            D[d_name] = model.addMVar(shape=n_delta, vtype=GRB.BINARY, name=d_name)
            P[p_name] = model.addVar(vtype=GRB.CONTINUOUS, name=p_name)

    for i, j in f.mdl.use_interaction_features:
        if (i in g.user_available_x) and (j in g.user_available_x): #2変数ともユーザが動かせる変数
            n_f0_th, n_f1_th = len(f.interaction_feature_area[(i, j)]['f0_threshold']), len(f.interaction_feature_area[(i, j)]['f1_threshold'])
            matrix = f.interaction_feature_area[(i, j)]['value_matrix']

            l0_name, l1_name, d0_name, d1_name, m_name, p_name = f"li{index}: ({i, j, i})", f"li{index}: ({i, j, j})", f"di{index}: ({i, j, i})", f"di{index}: ({i, j, j})", f"m{index}: ({i, j})", f"p{index}: ({i, j})"
            L[l0_name] = model.addMVar(shape=n_f0_th, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=l0_name) 
            L[l1_name] = model.addMVar(shape=n_f1_th, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=l1_name)    
            D[d0_name] = model.addMVar(shape=n_f0_th, vtype=GRB.BINARY, name=d0_name)     
            D[d1_name] = model.addMVar(shape=n_f1_th, vtype=GRB.BINARY, name=d1_name)
            M[m_name] =  model.addMVar(shape=matrix.shape[0] * matrix.shape[1], vtype=GRB.BINARY, name=m_name)
            P[p_name] =  model.addVar(vtype=GRB.CONTINUOUS, name=p_name)

        if (i in g.user_available_x) and (j in g.environment_s): #変数iがユーザが動かせる変数
            n_f0_th = len(f.interaction_feature_area[(i, j)]['f0_threshold'])

            l0_name, d0_name, p_name = f"li{index}: ({i, j, i})", f"di{index}: ({i, j, i})", f"p{index}: ({i, j})"
            L[l0_name] = model.addMVar(shape=n_f0_th, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=l0_name)    
            D[d0_name] = model.addMVar(shape=n_f0_th, vtype=GRB.BINARY, name=d0_name)
            P[p_name] =  model.addVar(vtype=GRB.CONTINUOUS, name=p_name)
        if (i in g.environment_s) and (j in g.user_available_x): #変数jがユーザが動かせる変数
            n_f1_th = len(f.interaction_feature_area[(i, j)]['f1_threshold'])

            l1_name, d1_name, p_name = f"li{index}: ({i, j, j})", f"di{index}: ({i, j, j})", f"p{index}: ({i, j})"
            L[l1_name] = model.addMVar(shape=n_f1_th, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"li{index}: ({i, j, j})")    
            D[d1_name] = model.addMVar(shape=n_f1_th, vtype=GRB.BINARY, name=f"di{index}: ({i, j, j})")
            P[p_name] =  model.addVar(vtype=GRB.CONTINUOUS, name=p_name)
    model.update()

    #ユーザが動かせる変数が絡む森に対してGurobi用変数を用いてMIP制約を課す．環境変数のみなら定数を計算
    for i in f.main_feature_area.keys():
        if i in g.environment_s: # 変数が環境変数なら
            for leaf in f.main_feature_area[i]:
                if s[i - g.n_user_avalable] <= leaf[1]:
                    pred_by_environment += leaf[2]
                    break
        else: # 変数がユーザが動かせる変数なら
            # 普通にMIP追加
            n_threshold = len(f.main_feature_area[i]['threshold'])
            n_delta = len(f.main_feature_area[i]['value'])
            l_name, d_name, p_name = f"l{index}: {i}", f"d{index}: {i}", f"p{index}: {i}"
            l, d, p = L[l_name], D[d_name], P[p_name]
            
            model.addConstr(l @ f.main_feature_area[i]['threshold'] == x[i])
            model.addConstr(d @ f.main_feature_area[i]['value'] == p)
            model.addConstr(l.sum() == 1)
            for th_i in range(n_threshold):
                if th_i == 0:
                    model.addConstr(l[th_i] - d[th_i] <= 0)
                elif th_i == n_threshold - 1:
                    model.addConstr(l[th_i] - d[th_i-1] <= 0)
                else:
                    model.addConstr(l[th_i] - d[th_i] - d[th_i-1] <= 0)
            model.addConstr(d.sum() == 1)
        sum_feature_mean += -f.mdl.train_main_mean[i]

    for i, j in f.interaction_feature_area.keys():
        area = f.interaction_feature_area[(i, j)]
        f0_th, f1_th, matrix = area['f0_threshold'], area['f1_threshold'], area['value_matrix']
        f0_index, f1_index = 0, 0
        
        if (i in g.environment_s) and (j in g.environment_s): #2変数とも環境変数
            for index in range(len(f0_th)):
                if s[i - g.n_user_avalable] <= f0_th[index]:
                    if s[i - g.n_user_avalable] != min_bound[i]:
                        f0_index = index - 1
                    break
            for index in range(len(f1_th)):
                if s[j - g.n_user_avalable] <= f1_th[index]:
                    if s[j - g.n_user_avalable] != min_bound[j]:
                        f1_index = index - 1
                    break
            pred_by_environment += matrix[f0_index][f1_index]
        elif (i in g.environment_s) and (j in g.user_available_x): #変数iが環境変数
            for index in range(len(f0_th)):
                if s[i - g.n_user_avalable] <= f0_th[index]:
                    if s[i - g.n_user_avalable] != min_bound[i]:
                        f0_index = index - 1
                    break

            n_threshold = len(f1_th)
            n_delta = len(matrix.shape[1])
            l1_name, d1_name, p_name = f"li{index}: ({i, j, j})", f"di{index}: ({i, j, j})",  f"p{index}: ({i, j})"
            l, d, p = L[l1_name], D[d1_name], P[p_name]
            
            model.addConstr(l @ f1_th == x[j])
            model.addConstr(d @ matrix[i, :] == p)
            model.addConstr(l.sum() == 1)
            for i in range(n_threshold):
                if i == 0:
                    model.addConstr(l[i] - d[i] <= 0)
                elif i == n_threshold - 1:
                    model.addConstr(l[i] - d[i-1] <= 0)
                else:
                    model.addConstr(l[i] - d[i] - d[i-1] <= 0)
            model.addConstr(d.sum() == 1)
        elif (i in g.user_available_x) and (j in g.environment_s): #変数jが環境変数
            for index in range(len(f1_th)):
                if s[j - g.n_user_avalable] <= f1_th[index]:
                    if s[j - g.n_user_avalable] != min_bound[j]:
                        f1_index = index - 1
                    break

            n_threshold = len(f0_th)
            l0_name, d0_name, p_name = f"li{index}: ({i, j, i})", f"di{index}: ({i, j, i})", f"p{index}: ({i, j})"
            l, d, p = L[l0_name], D[d0_name], P[p_name]
            
            model.addConstr(l @ f0_th == x[i])
            model.addConstr(d @ matrix[:, j] == p)
            model.addConstr(l.sum() == 1)
            for i in range(n_threshold):
                if i == 0:
                    model.addConstr(l[i] - d[i] <= 0)
                elif i == n_threshold - 1:
                    model.addConstr(l[i] - d[i-1] <= 0)
                else:
                    model.addConstr(l[i] - d[i] - d[i-1] <= 0)
            model.addConstr(d.sum() == 1)
        else:
            # 2つともdeltaを設定されている．deltaSを掛け算するように設定
            n_threshold0 = len(f0_th)
            n_threshold1 = len(f1_th)
            l0_name, l1_name, d0_name, d1_name, m_name, p_name = f"li{index}: ({i, j, i})", f"li{index}: ({i, j, j})", f"di{index}: ({i, j, i})", f"di{index}: ({i, j, j})", f"m{index}: ({i, j})", f"p{index}: ({i, j})"
            l0, l1, d0, d1, m, p = L[l0_name], L[l1_name], D[d0_name], D[d1_name], M[m_name], P[p_name]
            
            #i番目の変数がどの区域入っているかをワンホットで表す(d0)
            model.addConstr(l0 @ f0_th == x[i])
            model.addConstr(l0.sum() == 1)
            for k in range(n_threshold0):
                if k == 0:
                    model.addConstr(l0[k] - d0[k] <= 0)
                elif k == n_threshold0 - 1:
                    model.addConstr(l0[k] - d0[k-1] <= 0)
                else:
                    model.addConstr(l0[k] - d0[k] - d0[k-1] <= 0)
            model.addConstr(d0.sum() == 1)

            #j番目の変数がどの区域入っているかをワンホットで表す(d1)
            model.addConstr(l1 @ f1_th == x[j])
            model.addConstr(l1.sum() == 1)
            for k in range(n_threshold1):
                if k == 0:
                    model.addConstr(l1[k] - d1[k] <= 0)
                elif k == n_threshold1 - 1:
                    model.addConstr(l1[k] - d1[k-1] <= 0)
                else:
                    model.addConstr(l1[k] - d1[k] - d1[k-1] <= 0)
            model.addConstr(d1.sum() == 1)

            #i,jの変数が入っている区域の両方を満たすMだけが立ち上がるワンホット(M)
            for s_ in range(n_threshold0):
                for t in range(n_threshold1):
                    l0[s_] + l1[t] - 2 * m[s_ * n_threshold0 + t] >= 0
                    l0[s_] + l1[t] - m[s_ * n_threshold0 + t] <= 1
            values_1d = area.ravel()
            model.addConstr(values_1d @ m == p)
        sum_feature_mean += -f.mdl.train_interaction_mean[(i, j)]

    model.addConstr(quicksum(p for p in P.values()) == y)
    return model