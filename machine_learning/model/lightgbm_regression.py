from utility.module import *
from utility.constant import *

from machine_learning.model.base_model import *
from machine_learning.tools.metrics import *
from machine_learning.tools.tree_visualize import *


class MyLightGBM(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = LIGHTGBM
        self.gsr_flg = False
        self.bayse_flg = False
 

    def fit(self, x, y):
        if self.gsr_flg:
            self.mdl = lgb.LGBMRegressor(
                                            learning_rate = self.gsr.best_params_['learning_rate'],
                                            max_depth = self.gsr.best_params_['max_depth'],
                                            num_leaves = self.gsr.best_params_['num_leaves'],
                                            reg_alpha = self.gsr.best_params_['reg_alpha'],
                                            reg_lambda = self.gsr.best_params_['reg_lambda'],
                                            min_child_samples = self.gsr.best_params_['min_child_samples'],
                                            random_state = self.gsr.best_params_['random_state'])
        elif self.bayse_flg:
            self.mdl = lgb.LGBMRegressor(**self.best_params, n_estimators=500)
        else:
            self.mdl = lgb.LGBMRegressor()
        self.mdl.fit(x, y)

    def bayse_search(self, x, y):
        base_param = {
            'objective':'regression',
            'metric':'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': g.seed,
            'learning_rate': 0.1
            }
        searcher = BayseSearch(x, y, base_param)
        self.best_params = searcher.fit()
        self.bayse_flg = True

    def grid_search(self, x, y):
        self.search_params = {
            'learning_rate'     : [0.01, 0.05, 0.1], # [0.1], #[0.01, 0.05, 0.1],
            'max_depth'         : [5, 7, 9], #[5], #[5,7,9],
            'num_leaves'        : [7, 15, 31], #[100], #[7, 15, 31],
            'reg_alpha'         : [0, 2, 5], #[0.1], #[0, 2, 5],
            'reg_lambda'        : [0, 2, 5], # [0.1], #[0, 2, 5],
            'min_child_samples' : [20, 30, 50], # [20], #[20, 30, 50]
            'random_state'      : [g.seed]
        }
        kf = KFold(n_splits=n_split)
        self.gsr = GridSearchCV(
            lgb.LGBMRegressor(),
            self.search_params,
            scoring=make_scorer(RMSE, greater_is_better=False),
            cv = kf,
            n_jobs = -1,
            verbose=3
        )
        self.gsr.fit(x, y)
        self.gsr_flg = True


class BayseSearch():
    def __init__(self, data_train, label_train, base_param):
        self.data_train = data_train
        self.label_train = label_train
        self.base_param = base_param

    def objective(self, trial):
        params = {
            'num_leaves':trial.suggest_int('num_leaves', 3, 400), # 200
            'max_depth':trial.suggest_int('max_depth', 3, 12),
            'reg_alpha':trial.suggest_float('reg_alpha', 1e-5, 3.0),
            'reg_lambda':trial.suggest_float('reg_lambda', 1e-5, 3.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'subsample':trial.suggest_float('subsample', 0.1, 1.0),
            'subsample_freq':trial.suggest_int('subsample_freq', 0, 5),
            'min_child_samples':trial.suggest_int('min_child_samples', 1, 100),
            }
        params.update(self.base_param)
        
        model = lgb.LGBMRegressor(**params, n_estimators=500)
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