from utility.module import *
from utility.constant import *
from utility.setting import *
import utility.gloabl_values as g

from machine_learning.tools.metrics import *
from machine_learning.model.base_model import *

class MySupportVectorRegression(BaseMyModel):
    def __init__(self, i, svr_kernel_name) -> None:
        super().__init__(i)
        self.name = svr_kernel_name
        self.gsr_flg = False
        self.bayse_flg = False 

    def fit(self, x, y):
        if self.gsr_flg:
            print(self.gsr.best_params_)
            self.mdl = svm.SVR(**self.gsr.best_params_)
        elif self.bayse_flg:
            self.mdl = svm.SVR(**self.best_params)
        else:
            self.mdl = svm.SVR()
        self.mdl.fit(x, y)
        self.x_train_var = np.var(x)

    def bayse_search(self, x, y):
        base_param = {
            'kernel': search_params[self.name]['kernel'][0]
        }
        searcher = BayseSearch(x, y, base_param)
        self.best_params = searcher.fit()
        self.bayse_flg = True
    
    def grid_search(self, x, y):
        kf = KFold(n_splits=n_split)
        self.gsr = GridSearchCV(
            svm.SVR(),
            search_params[self.name],
            scoring=make_scorer(RMSE, greater_is_better=False),
            return_train_score=True,
            n_jobs=-1,
            cv = kf,
            verbose=3
        )
        self.gsr.fit(x, y)
        self.gsr_flg = True

    def set_parameter(self):
        self.sv = np.array(self.mdl.support_vectors_)
        self.n_sv = len(self.sv)
        self.coef = self.mdl.dual_coef_[0]
        self.ic = self.mdl.intercept_[0]
        self.gamma = 1 / (g.n_feature * self.x_train_var)
        self.degree = self.mdl.degree
        self.coef0 = self.mdl.coef0

    def set_parameter_for_opt(self, s):
        if g.select_ml == SVRLINEAR:
            calced_list = []
            for i in range(self.n_sv):
                sum_value = 0
                for j in range(g.n_environment_s):
                    s_total_index = j + g.n_user_available_x
                    sum_value += s[j] * self.sv[i][s_total_index]
                calced_list.append(sum_value)
            self.ic_by_kernel = calced_list

    def predict(self, x):
        return self.mdl.predict(x)

    def predict_manual(self, x):
        kernel_value = []
        if self.name == SVRLINEAR:
            for i in range(self.n_sv):
                sum_value = 0
                for j in range(len(x)):
                    sum_value += x[j] * self.sv[i][j]
                kernel_value.append(sum_value)
        elif self.name == SVRPOLY:
            for i in range(self.n_sv):
                sum_value = 0
                for j in range(len(x)):
                    sum_value += x[j] * self.sv[i][j]
                sum_value *= self.gamma
                sum_value += self.coef0
                sum_value = sum_value ** self.degree
                kernel_value.append(sum_value)
        elif self.name == SVRGAUSS:
            for i in range(self.n_sv):
                exp_value = 0
                for j in range(len(x)):
                    exp_value += (x[j] - self.sv[i][j]) ** 2
                kernel_value.append(np.exp(-self.gamma * (exp_value)))
        ans = np.sum(np.array([self.coef[i] * kernel_value[i] for i in range(self.n_sv)])) + self.ic
        return ans


class BayseSearch():
    def __init__(self, data_train, label_train, base_param):
        self.data_train = data_train
        self.label_train = label_train
        self.base_param = base_param

    def objective(self, trial):
        params = {
            'C': trial.suggest_float('C', 0, 3),
            'epsilon': trial.suggest_float('epsilon', 0, 3)
        }
        if g.select_ml == SVRPOLY:
            params['degree'] = trial.suggest_int('degree', 2, 10)

        params.update(self.base_param)
        
        model = svm.SVR(**params)
        kf = KFold(n_splits=n_split, shuffle=True, random_state=g.seed)
        scores = cross_validate(model, X=self.data_train, y=self.label_train, scoring=make_scorer(RMSE), cv=kf)
        print(f"val score:{scores['test_score'].mean()}")
        return scores['test_score'].mean()

    def fit(self):
        study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=g.seed))
        study.optimize(self.objective, n_trials=n_trial)

        best_params = study.best_params
        
        params = self.base_param
        params.update(best_params)
        print(study.best_value)
        return params


search_params = {}
search_params[SVRLINEAR] = {
    'kernel':["linear"],
    'C' : [0.1, 0.5, 1],#正則化パラメーター。正則化の強さはCに反比例します。厳密に正でなければなりません。ペナルティは、l2ペナルティの2乗です。
    'epsilon': [0.1, 0.25, 0.5, 0.75, 1],#イプシロン-SVRモデルのイプシロン。これは、トレーニング損失関数でペナルティが実際の値からイプシロンの距離内で予測されたポイントに関連付けられていないイプシロンチューブを指定します。
}
search_params[SVRPOLY] = {
    'kernel':["poly"],
    'degree':[2, 3, 4, 5],
    'C' : [0.5, 1, 3, 5],#正則化パラメーター。正則化の強さはCに反比例します。厳密に正でなければなりません。ペナルティは、l2ペナルティの2乗です。
    'epsilon': [0.1, 0.5, 1, 3],#イプシロン-SVRモデルのイプシロン。これは、トレーニング損失関数でペナルティが実際の値からイプシロンの距離内で予測されたポイントに関連付けられていないイプシロンチューブを指定します。
}
search_params[SVRGAUSS] = {
    'kernel':["rbf"],
    'degree':[1], 
    'C' : [0.1, 0.5, 1, 2, 3, 4, 5],#正則化パラメーター。正則化の強さはCに反比例します。厳密に正でなければなりません。ペナルティは、l2ペナルティの2乗です。
    'epsilon': [0, 0.1, 0.5, 1, 2, 3],#イプシロン-SVRモデルのイプシロン。これは、トレーニング損失関数でペナルティが実際の値からイプシロンの距離内で予測されたポイントに関連付けられていないイプシロンチューブを指定します。
}
