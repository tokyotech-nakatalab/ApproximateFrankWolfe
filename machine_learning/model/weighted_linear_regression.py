from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *

class MyWeightedLinearRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = LINEARREGRESSION

    def set_data(self, x, y):
        self.x = add_intercept(x)
        self.y = y

    def fit_xk(self, x_k):
        x_k = add_intercept(x_k)
        X = self.x
        y = self.y
        W = np.diag(np.array([kernel(x_i, x_k, self.weight_sigma) for x_i in X]))
        W[W < 10**(-10)] = 0.
        try:
            a = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        except:
            print("特異行列なので疑似逆行列を求めます")
            a = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        self.coef_ = a

    def set_parameter(self):
        try:
            self.linear_coef = self.coef_[1:]
            self.linear_ic = self.coef_[0]
        except:
            return

    def set_parameter_for_opt(self, s):
        ic = self.linear_coef[g.n_user_available_x:] @ s
        self.coefs = self.linear_coef[:g.n_user_available_x]
        self.ic = ic + self.linear_ic

    def predict(self, xs):
        return xs @ self.linear_coef + self.linear_ic

    def eval_performance(self, x_train, y_train, x_test, y_test):
        y_hat = self.good_obj_mdl.predict(x_train)
        tr_rmse = RMSE(y_train, y_hat)
        print('Train RMSE : {:.3f}'.format(tr_rmse))

        y_hat = self.good_obj_mdl.predict(x_test)
        ts_rmse = RMSE(y_test, y_hat)
        print('Test RMSE : {:.3f}'.format(ts_rmse))
        return tr_rmse, ts_rmse
        
    def init_weight_sigma(self):
        self.weight_sigma = self.good_obj_mdl.mdl.weight_sigma

    def set_weight_sigma(self, sigma):
        self.weight_sigma = sigma

class MyLocalLinearRegression(BaseMyModel):
    def __init__(self, i, max_bound, min_bound) -> None:
        super().__init__(i)
        self.name = WEIGHTEDLINEARREGRESSION
        self.bayse_flg = False 
        self.base_param = {
            'max_bound': max_bound,
            'min_bound': min_bound
        }

    def bayse_search(self, x, y):
        searcher = BayseSearch(x, y, self.base_param)
        self.best_params = searcher.fit()
        self.bayse_flg = True

    def fit(self, x, y):
        if self.bayse_flg:
            self.mdl = LocalLinearRegression(**self.best_params)
        else:
            params = self.base_param
            params['weight_sigma'] = ((self.base_param['max_bound'] - self.base_param['min_bound']) ** g.n_feature) / 128 * 2
            print(params['weight_sigma'])
            self.mdl = LocalLinearRegression(**params)
        self.mdl.fit(x, y)

    def predict(self, x):
        if x.ndim == 1:
            return self.mdl.predict(x.reshape(1, g.n_feature))
        else:
            return self.mdl.predict(x)


class LocalLinearRegression():
    def __init__(self, weight_sigma, max_bound, min_bound) -> None:
        self.weight_sigma = weight_sigma
        self.n_feature_particle = n_feature_particle
        self.max_bound, self.min_bound = max_bound, min_bound
        self.linspace_feature = np.round(np.linspace(self.min_bound, self.max_bound, self.n_feature_particle), 4)
        self.predict_step = (self.max_bound - self.min_bound) / (self.n_feature_particle - 1)
        self.predict_harf_step = self.predict_step / 2

    def set_data(self, x, y):
        self.x = add_intercept(x)
        self.y = y

    def fit(self, x, y):
        self.set_data(x, y)

        p = itertools.permutations(np.tile(self.linspace_feature, g.n_feature), g.n_feature)
        self.points = np.array(sorted(list(set(p))))
        self.point_coef = {tuple(self.points[i]): None for i in range(self.points.shape[0])}
        for xk in self.points:
            coef_ = self.fit_xK(np.array(xk))
            self.point_coef[tuple(xk)] = {}
            self.point_coef[tuple(xk)]['coef'] = coef_[1:]
            self.point_coef[tuple(xk)]['ic'] = coef_[0]

    def fit_xK(self, x_k):
        x_k = add_intercept(x_k)
        X = self.x
        y = self.y
        W = np.diag(np.array([kernel(x_i, x_k, self.weight_sigma) for x_i in X]))
        W[W < 10**(-10)] = 0.
        try:
            a = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        except:
            print("特異行列なので疑似逆行列を求めます")
            a = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        return a
        
    def predict(self, xs):
        n_data = xs.shape[0]
        near_xk = self.near_point(xs)
        near_xk = [tuple(near_xk[i]) for i in range(n_data)]
        point_ic = np.array([self.point_coef[near_xk[i]]['ic'] for i in range(n_data)])
        point_coef = np.array([self.point_coef[near_xk[i]]['coef'] for i in range(n_data)])
        preds = []
        for i in range(n_data):
            preds.append(xs[i] @ point_coef[i] + point_ic[i])
        if n_data != 1:
            return np.array(preds)
        else:
            return preds[0]

    def near_point(self, x):
        near_p = np.full_like(x, -np.inf)
        n_step = (x - self.min_bound) // self.predict_step
        near_p[n_step * self.predict_step + self.predict_harf_step < x] = ((n_step + 1) * self.predict_step)[n_step * self.predict_step + self.predict_harf_step < x]
        near_p[near_p == -np.inf] = (n_step * self.predict_step)[near_p == -np.inf]
        near_p += self.min_bound
        return np.round(near_p, 4)



class BayseSearch():
    def __init__(self, data_train, label_train, base_param):
        self.data_train = data_train
        self.label_train = label_train
        self.base_param = base_param

    def objective(self, trial):
        params = {
            'weight_sigma':trial.suggest_uniform('weight_sigma', 0.1, 2.0)
            }
        params.update(self.base_param)
        
        model = LocalLinearRegression(**params)
        aves = []
        kf = KFold(n_splits=n_split, shuffle=True, random_state=g.seed)
        for train_index, test_index in kf.split(self.data_train, self.label_train):
            model.fit(self.data_train[train_index], self.label_train[train_index])
            y_hat = model.predict(self.data_train[test_index])
            aves.append(RMSE(self.label_train[test_index], y_hat))
        score = sum(aves) / n_split
        return score

    def fit(self):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=g.seed))
        study.optimize(self.objective, n_trials=n_trial)

        best_params = study.best_params
        
        params = self.base_param
        params.update(best_params)
        print(study.best_value)
        return params
