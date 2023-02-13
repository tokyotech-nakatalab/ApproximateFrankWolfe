from utility.setting import *
from utility.module import *
from utility.constant import *


def listtree2rule(tree, min_x, max_x):
    node_indexes = node2index(tree)
    rules = []
    left = [node_indexes[node[node_info['left_child']]] for node in tree]
    right = [node_indexes[node[node_info['right_child']]] for node in tree]
    threshold = [node[node_info['threshold']] for node in tree]
    values = [node[node_info['value']] for node in tree]

    def get_rule(left, right, threshold, rule, node):
        if left[node] != -1:
            my_rule = rule.copy()
            my_rule[1] = threshold[node]
            if np.isnan(threshold[left[node]]):
                my_rule[2] = values[left[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, my_rule, left[node])
        if right[node] != -1:
            my_rule = rule.copy()
            my_rule[0] = threshold[node]
            if np.isnan(threshold[right[node]]):
                my_rule[2] = values[right[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, rule, right[node])

    get_rule(left, right, threshold, [min_x, max_x, -1], 0)
    for i in range(1, len(rules) - 1):
        if rules[i][0] == min_x:
            rules[i][0] = rules[i-1][1]
        if rules[i][1] == max_x:
            rules[i][1] = rules[i+1][0]
    # for rule in rules:
    #     print(f"if {rule[0]} < x <= {rule[1]}:")
    #     print(f"     {rule[2]}")
    # print("*****************************")
    return copy.deepcopy(rules)


def listtree2rule_interaction(tree, min_x, max_x):
    node_indexes = node2index(tree)
    feature_ids =  {"Column_0": 0, "Column_1": 1, None: -1}
    rules = []
    left = [node_indexes[node[node_info['left_child']]] for node in tree]
    right = [node_indexes[node[node_info['right_child']]] for node in tree]
    threshold = [node[node_info['threshold']] for node in tree]
    values = [node[node_info['value']] for node in tree]
    feature = [feature_ids[node[node_info['split_feature']]] for node in tree]

    def get_rule(left, right, threshold, feature, rule, node):
        rule_feature = feature[node]
        if left[node] != -1:
            my_rule = rule.copy()
            my_rule[rule_feature * 2 + 1] = threshold[node]
            if np.isnan(threshold[left[node]]):
                my_rule[4] = values[left[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, feature, my_rule, left[node])
        if right[node] != -1:
            my_rule = rule.copy()
            my_rule[rule_feature * 2] = threshold[node]
            if np.isnan(threshold[right[node]]):
                my_rule[4] = values[right[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, feature, my_rule, right[node])

    get_rule(left, right, threshold, feature, [min_x, max_x, min_x, max_x, -1], 0)
    # for i in range(1, len(rules) - 1):
    #     if rules[i][0] == min_x:
    #         rules[i][0] = rules[i-1][1]
    #     if rules[i][1] == max_x:
    #         rules[i][1] = rules[i+1][0]
    # for i in range(1, len(rules) - 1):
    #     if rules[i][2] == min_x:
    #         rules[i][2] = rules[i-1][3]
    #     if rules[i][3] == max_x:
    #         rules[i][3] = rules[i+1][2]

    # for rule in rules:
    #     print(f"if {rule[0]} < x0 <= {rule[1]} & {rule[2]} < x1 <= {rule[3]}:")
    #     print(f"     {rule[4]}")
    # print("*****************************")
    return copy.deepcopy(rules)
        


def node2index(tree):
    dic = {}
    for i, node in enumerate(tree):
        dic[node[node_info["node_index"]]] = i
    dic[None] = -1
    return dic

def forest2trees(forest_nodes):
    trees = []
    tree = []
    now_t = 0
    for node in forest_nodes:
        if now_t != node[0]:
            trees.append(tree)
            tree = []
            now_t = node[0]
        tree.append(node)
    trees.append(tree)
    return trees
        

def dt2rule(tree, min_x, max_x):
    '''
    Converting scikit-learn's DecisionTreeClassifier to Python code

    Args:
        <sklearn.tree.DecisionTreeClassifier> tree
        <list> feature_names
        <list> class_names
        <str> func_name
    Return:
        <list> rule
    '''
    rules = []
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = [v[0][0] for v in tree.tree_.value]
    n_node_samples = tree.tree_.n_node_samples

    def get_rule(left, right, threshold, rule, node):
        global rules
        if left[node] != -1:
            my_rule = rule.copy()
            my_rule[1] = threshold[node]
            if threshold[left[node]] == -2:
                my_rule[2] = value[left[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, my_rule, left[node])
        if right[node] != -1:
            my_rule = rule.copy()
            my_rule[0] = threshold[node]
            if threshold[right[node]] == -2:
                my_rule[2] = value[right[node]]
                rules.append(my_rule)
            else:
                get_rule(left, right, threshold, rule, right[node])

    get_rule(left, right, threshold, [min_x, max_x, -1], 0)
    for i in range(1, len(rules) - 1):
        if rules[i][0] == min_x:
            rules[i][0] = rules[i-1][1]
        if rules[i][1] == max_x:
            rules[i][1] = rules[i+1][0]
    # for rule in rules:
    #     print(f"if {rule[0]} < x <= {rule[1]}:")
    #     print(f"     {rule[2]}")
    # print("*****************************")
    return copy.deepcopy(rules)