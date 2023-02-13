from utility.module import *
from machine_learning.tools.metrics import *
from machine_learning.tools.tree_visualize import *
from utility.constant import *


class MyRandomForest():
    def __init__(self) -> None:
        self.name = RANDOMFOREST
        search_params = {
            'n_estimators'      : [1], #[5, 10, 20, 30, 50, 100, 300],
            'max_features'      : [1], # [i for i in range(1, d)],
            'random_state'      : [2525],
            'n_jobs'            : [1],
            'min_samples_split' : [3], #[3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            'max_depth'         : [5]  #[3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
        }
        
        self.gsr = GridSearchCV(
            RandomForestRegressor(),
            search_params,
            scoring=make_scorer(RMSE,greater_is_better=False),
            cv = 5,
            n_jobs = -1,
            verbose=True
        )


    def fit(self, x, y):
        self.gsr.fit(x, y)
        self.mdl = RandomForestRegressor(n_estimators = self.gsr.best_params_['n_estimators'],
                                         max_features = self.gsr.best_params_['max_features'],
                                         random_state = self.gsr.best_params_['random_state'],
                                         n_jobs = self.gsr.best_params_['n_jobs'],
                                         min_samples_split = self.gsr.best_params_['min_samples_split'],
                                         max_depth = self.gsr.best_params_['max_depth'])
        self.mdl.fit(x, y)

    def set_parameter(self):
        union_tree = self.forest2uniontree()
        threshold = []
        pred = []
        for i, leaf in enumerate(union_tree):
            if i == 0:
                threshold.append(leaf[0])
            threshold.append(leaf[1])
            pred.append(leaf[2])
        self.threshold = np.array(threshold)
        self.pred = np.array(pred)

        # test_x = 2.1
        # for leaf in union_tree:
        #     if test_x <= leaf[1]:
        #         print(leaf[2])
        #         break



    def forest2uniontree(self):
        forests = [dt2rule(self.mdl.estimators_[i], min_x, max_x) for i in range(len(self.mdl.estimators_))]
        leaf_index = [0] * len(forests)
        union_tree = []
        lower = min_x
        while True:
            upper_bounds = np.array([forests[i][leaf_index[i]][1] for i in range(len(forests))])
            leaf_values = np.array([forests[i][leaf_index[i]][2] for i in range(len(forests))])
            min_tree_i = np.argmin(upper_bounds)

            new_rule = [lower, upper_bounds[min_tree_i], np.average(leaf_values)]
            union_tree.append(new_rule)
            leaf_index[min_tree_i] += 1
            if sum([forests[i][leaf_index[i]][1] for i in range(len(forests))]) == max_x * len(forests):
                break
        
        # test_x = 2.1
        # for leaf in union_tree:
        #     if test_x <= leaf[1]:
        #         print(leaf[2])
        #         break

        # ans = 0
        # for forest in forests:
        #     for leaf in forest:
        #         if test_x <= leaf[1]:
        #             ans += leaf[2]
        #             break
        # ans /= len(forests)
        # print(ans)
        return union_tree