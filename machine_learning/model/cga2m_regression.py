from utility.module import *
from utility.constant import *
from utility.setting import *

from machine_learning.tools.metrics import *
from machine_learning.tools.tree_visualize import *
from machine_learning.model.base_model import *

class MyCGA2M(BaseMyModel):
    def __init__(self, i, X_train, y_train, X_eval, y_eval) -> None:
        self.no = i
        self.name = CGA2M
        super().__init__(i)

        # parameters of LightGBM in CGA2M+
        base_params = {'objective': 'regression',
                    'metric': 'rmse',
                    'verbose': -1,
                    'num_iteration': 100}
        lgbm_params = {'num_leaves': 10}
        exec_params = {**base_params, **lgbm_params}

        self.feature_combination = list(itertools.combinations(range(X_train.shape[1]), 2))
        self.max_outer_iteration = len(self.feature_combination)
        print("max_outer_iteration=", self.max_outer_iteration)

        self.mdl = Constraint_GA2M(X_train,
                                   y_train,
                                   X_eval,
                                   y_eval,
                                   exec_params,
                                   monotone_constraints = None, #[1,0,0,0,0,0,0],# monotonic constraints
                                   all_interaction_features = self.feature_combination)

    def fit(self):
        self.mdl.train(max_outer_iteration=self.max_outer_iteration, backfitting_iteration=20, threshold=0.05)
        print('START prune_and_retrain')
        self.mdl.prune_and_retrain(threshold=0.05,backfitting_iteration=30)
        #self.mdl.higher_order_train()


    def predict(self, x):
        #preds_higher = cga2m_no1.predict(X_test,higher_mode=True)
        preds = self.mdl.predict(x)
        return preds

    def set_parameter(self):
        self.main_feature_area, self.interaction_feature_area = {}, {}
        area_rules_by_main_feature, area_rules_by_interaction_feature = self.forest2uniontree()
        for i in area_rules_by_main_feature.keys():
            threshold = []
            pred = []
            for leaf_i, leaf in enumerate(area_rules_by_main_feature[i]):
                if leaf_i == 0:
                    threshold.append(leaf[0])
                threshold.append(leaf[1])
                pred.append(leaf[2])
            self.main_feature_area[i] = {'threshold': np.array(threshold), 'value': np.array(pred)}
        for i, j in area_rules_by_interaction_feature.keys():
            area = area_rules_by_interaction_feature[(i, j)]
            f0_th, f1_th, matrix = area[0], area[1], area[2]
            self.interaction_feature_area[(i, j)] = {'f0_threshold': f0_th, 'f1_threshold': f1_th, "value_matrix": np.array(matrix)}

        # test_x = 2.1
        # for leaf in union_tree:
        #     if test_x <= leaf[1]:
        #         print(leaf[2])
        #         break

    def set_parameter_for_opt(self, s):
        pass


    def forest2uniontree(self):
        area_rules_by_main_feature = {}
        area_rules_by_interaction_feature = {}
        for feature_id in self.mdl.main_model_dict.keys():
            feature_forest = self.mdl.main_model_dict[feature_id]
            rule_forest = []
            #list_forests = (feature_forest.trees_to_dataframe()).columns.tolist()
            list_forests = (feature_forest.trees_to_dataframe()).values.tolist()
            list_trees = forest2trees(list_forests)
            for tree in list_trees:
                rule_tree = listtree2rule(tree, min_xs, max_xs)
                rule_forest.append(rule_tree)
            leaf_index = [0] * len(rule_forest)
            union_tree = []
            lower_bound = min_xs
            next_fin_flg = False
            while True:
                upper_bounds = np.array([rule_forest[i][leaf_index[i]][1] for i in range(len(rule_forest))])
                leaf_values = np.array([rule_forest[i][leaf_index[i]][2] for i in range(len(rule_forest))])
                min_upper_bound = np.min(upper_bounds)

                new_rule = [lower_bound, min_upper_bound, np.sum(leaf_values)]
                union_tree.append(new_rule)
                lower_bound = np.min(min_upper_bound)
                for i in range(len(upper_bounds)):
                    if upper_bounds[i] == min_upper_bound:
                        leaf_index[i] += 1
                if next_fin_flg:
                    break
                if sum([rule_forest[i][leaf_index[i]][1] for i in range(len(rule_forest))]) == max_xs * len(rule_forest):
                    next_fin_flg = True
            area_rules_by_main_feature[feature_id] = union_tree
        for feature_id in self.mdl.interaction_model_dict.keys():
            feature_forest = self.mdl.interaction_model_dict[feature_id]
            rule_forest = []
            list_forests = (feature_forest.trees_to_dataframe()).values.tolist()
            list_trees = forest2trees(list_forests)
            for tree in list_trees:
                rule_tree = listtree2rule_interaction(tree, min_xs, max_xs)
                rule_forest.append(rule_tree)
            union_tree = []
            f0_dic, f1_dic = {}, {}
            multi_f0_dic, multi_f1_dic = {}, {}
            counter = 0
            for rule_tree in rule_forest:
                for rule in rule_tree:
                    f0_dic[rule[0]], f0_dic[rule[1]], f1_dic[rule[2]], f1_dic[rule[3]] = 0, 0, 0, 0
                    if rule[0] == rule[1]:
                        counter += 1
                        if not rule[0] in multi_f0_dic:
                            multi_f0_dic[rule[0]] = 1
                        else:
                            multi_f0_dic[rule[0]] += 1
                    if rule[2] == rule[3]:
                        counter += 1
                        if not rule[2] in multi_f1_dic:
                            multi_f1_dic[rule[2]] = 1
                        else:
                            multi_f1_dic[rule[2]] += 1                            
            S = [[0] * (len(f1_dic.keys()) + len(multi_f1_dic.keys()) -1) for _ in range(len(f0_dic.keys()) + len(multi_f0_dic.keys()) - 1)]
            f0_dic = dict(sorted(f0_dic.items(), key=lambda x:x[0]))
            f1_dic = dict(sorted(f1_dic.items(), key=lambda x:x[0]))
            multi_f0 = sorted(multi_f0_dic.keys())
            multi_f1 = sorted(multi_f1_dic.keys())
            f0_i, f1_i = 0, 0
            for i, key in enumerate(f0_dic.keys()):
                f0_dic[key] = i + f0_i
                if len(multi_f0) != 0 and f0_dic[key] == multi_f0[f0_i]:
                    f0_i += 1
            for i, key in enumerate(f1_dic.keys()):
                f1_dic[key] = i + f1_i
                if len(multi_f1) != 0 and f1_dic[key] == multi_f1[f1_i]:
                    f1_i += 1
            for rule_tree in rule_forest:
                for rule in rule_tree:
                    f0_lower, f0_upper, f1_lower, f1_upper = f0_dic[rule[0]], f0_dic[rule[1]], f1_dic[rule[2]], f1_dic[rule[3]]
                    value = rule[4]
                    for i in range(f0_lower, f0_upper):
                        for j in range(f1_lower, f1_upper):
                            S[i][j] += value
            matrix_rule = [list(f0_dic.keys()), list(f1_dic.keys()), S]
            area_rules_by_interaction_feature[feature_id] = matrix_rule

        # 値のテスト
        for val_i in range(7):
            test_x = min_xs + np.random.random(7) * (max_xs - min_xs)

            preds = 0
            for i in self.mdl.use_main_features:
                preds_main = 0
                tree = area_rules_by_main_feature[i]
                for leaf in tree:
                    if test_x[i] <= leaf[1]:
                        preds_main += leaf[2]
                        break
                preds += preds_main - self.mdl.train_main_mean[i]
            for i, j in self.mdl.use_interaction_features:
                preds_interaction = 0
                tree = area_rules_by_interaction_feature[(i, j)]
                f0_th = tree[0]
                f1_th = tree[1]
                matrix = tree[2]
                f0_index = 0
                f1_index = 0
                for index in range(len(f0_th)):
                    if test_x[i] <= f0_th[index]:
                        if test_x[i] != min_xs:
                            f0_index = index - 1
                        break
                for index in range(len(f1_th)):
                    if test_x[j] <= f1_th[index]:
                        if test_x[j] != min_xs:
                            f1_index = index - 1
                        break
                preds_interaction = matrix[f0_index][f1_index]
                preds += preds_interaction - self.mdl.train_interaction_mean[(i, j)]
            preds += self.mdl.y_train_mean
            print(preds)


            preds = 0
            for i in self.mdl.use_main_features:
                preds_main = self.mdl.main_model_dict[i].predict(
                    copy.deepcopy(np.array([[test_x[i]]])).reshape(-1, 1),
                    num_iteration=self.mdl.main_model_dict[i].best_iteration,
                )
                preds = preds + preds_main - self.mdl.train_main_mean[i]

            for i, j in self.mdl.use_interaction_features:
                preds_interaction = self.mdl.interaction_model_dict[(i, j)].predict(
                    copy.deepcopy(np.array([[test_x[i], test_x[j]]])),
                    num_iteration=self.mdl.interaction_model_dict[(i, j)].best_iteration,
                )
                preds = preds + preds_interaction - self.mdl.train_interaction_mean[(i, j)]
            preds += self.mdl.y_train_mean
            print(preds)
        return area_rules_by_main_feature, area_rules_by_interaction_feature