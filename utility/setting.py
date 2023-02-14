from utility.constant import *
import utility.gloabl_values as g

""" 重要な実験設定 """
# 最適化したい問題
problem_list = [NONCONSTRAINT, BOXCONSTRAINT, DISKCONSTRAINT]
problem_list = [DISKCONSTRAINT]
problem_list = [NONCONSTRAINT]
# problem_list = [INEQUALITYCONSTRAINT]

# データの形状
data_type_list = [SINX01MOUNT2, ROSENBROCK, ACKELY]
data_type_list = [XSQUARE, MOUNT2]
# data_type_list = [ROSENBROCK]
# data_type_list = [SINX01MOUNT2]
data_type_list = [SINX0MOUNT2]
# data_type_list = [ACKELY]
data_type_list = [XSQUARE]
# # data_type_list = [COMPLEX7]
# data_type_list = [MOUNT2]
# data_type_list = [SINX0]

# 最適化手法
# opt_list = [TRUSTREGION]
opt_list = [MATHMATICALOPTIMIZATION, SIMULATEDANNEALING, FRANKWOLFE, SELECTBESTDATA]
# opt_list = [SIMULATEDANNEALING, FRANKWOLFE, FRANKWOLFE2, SELECTBESTDATA]
# opt_list = [FRANKWOLFE2, SELECTBESTDATA]
opt_list = [FRANKWOLFE, SELECTBESTDATA]
opt_list = [FRANKWOLFE]
# opt_list = [SIMULATEDANNEALING]
# opt_list = [FRANKWOLFE2]
# opt_list = [MATHMATICALOPTIMIZATION]
# opt_list = [SELECTBESTDATA]

# 機械学習手法
ml_list = [POLYNOMIALREGRESSION, LIGHTGBM, KNNLINEARREGRESSION]
ml_list = [POLYNOMIALREGRESSION, LIGHTGBM, ANNLINEARREGRESSION]
# ml_list = [LIGHTGBM, WEIGHTEDLINEARREGRESSION]
# ml_list = [LIGHTGBM]
# # # ml_list = [SVRGAUSS]
# # # ml_list = [SVRGAUSS, LIGHTGBM, ANNLINEARREGRESSION]
# ml_list = [ANNLINEARREGRESSION]
# ml_list = [KNNLINEARREGRESSION]
# ml_list = [NEURALNETWORK, SVRGAUSS, LIGHTGBM, LINEARREGRESSION]
# ml_list = [WEIGHTEDLINEARREGRESSION]
# ml_list = [CGA2M]
# ml_list = [LIGHTGBM, NEURALNETWORK]
# ml_list = [POLYNOMIALREGRESSION]


# シード
seed_list = list(range(10))
seed_list = list(range(5))
seed_list = [0]
# seed_list = list(range(3))
# seed_list = [4]
val_list = list(range(1))

""" 比較の為に変更するパラメータ """
n_item_list = list(range(1, 4))
n_item_list = [1]
# n_data_list = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
n_data_list = [200, 1000]
# n_data_list = [1000]
# n_data_list = [50]
# n_data_list = [2500]
# n_data_list = [100]
n_feature_list = [1, 5, 10, 30]
# n_feature_list = [1]
n_feature_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# n_feature_list = [5, 10, 15]
# n_feature_list = [30, 50]
# n_feature_list = [5]
# n_feature_list = [10]
# n_feature_list = [1]
n_user_available_x_list = n_feature_list

noise_sigma_list = list(range(0, 5))
noise_sigma_list = [1, 2, 3]
noise_sigma_list = [5]
# noise_sigma_list = [3]
# noise_sigma_list = [3]
n_nearest_list = list(range(10, 1010, 10))
# n_nearest_list = [400]
# n_nearest_list = list(range(10, 110, 10))
# n_nearest_list = list(range(10, 310, 10))
# n_nearest_list = [10, 20, 30]
# noise_sigma_list = [2]
# noise_sigma_list = [1]

# n_item_list = list(range(1, 2))
# n_data_list = [100, 500]
# n_feature_list = [1, 2]
# n_user_available_x_list = [1, 2]

# weight_sigma_list = [0.1]


""" 固定するパラメータ """
n_split = 5
train_rate = 0.8
eval_rate = 0.2
n_trial = 10
n_val_data = 100
n_feature_particle = 25
n_max_iteration = 1000
finish_epsilon = 0.25



""" 実験上の設定"""
is_only_learning = False # 学習のみを行うか
is_integrate_ml = False # 予測器を一つにまとめるか
already_optimize_ok = False # 最適化を再度行うか
is_only_shape_check = False # 形状の出力のみを行うか
same_val_data_size = True # valデータを固定数生成するか
search_hyper_paramerter = False # ハイパラ調整を行うか
only_user_available_x = True # ユーザの動かせる変数のみに絞るか
only_appropriate_feature = False # 無駄な特徴量の存在するパターンは排除するか
random_initialize_x = False # ランダムに初期解を選ぶか．Falseなら最もデータ点が高い位置からスタート
is_plot_proposed_method = False # 提案手法の特性についてプロットするか
force_minimize = True # 強制的に最小化問題に結果を置き換えるか
use_real_data = False # 広告データを使うか否か
is_x_normal = True # xを正規分布に従って発生させるか
visualize_optimization_process_status = DONTSEE # 最適化途中の可視化
auto_n_nearest = False

is_taisu = False # 片対数グラフをプロットするか
use_errorbar = False # エラーバーにするか散布図にするか



"""その他の設定"""
alpha = 0.5
min_xs = -1 #0.
max_xs = 1 # 300.
min_xs = -5
max_xs = 5
min_xs = -10
max_xs = 10
# min_xs = 0
# max_xs = 10
base_th_distance = 1.5
delta = 0.1

x_scale = 1
n_nearest_best_rate = 0.4


#実験1
# zikken_id = 1
# already_optimize_ok = False # 最適化を再度行うか
# n_data_list = [1000]
# n_feature_list = [10]
# n_nearest_list = list(range(10, 1010, 10))
# # n_nearest_list = [10]
# auto_n_nearest = False
# is_x_normal = True # xを正規分布に従って発生させるか
# x_scale = 3
# is_taisu = True
# opt_list = [FRANKWOLFE, SELECTBESTDATA]
# opt_list = [FRANKWOLFE]
# data_type_list = [XSQUARE]
# plot_name_dic[(solvers_names[FRANKWOLFE], mlmodel_names[KNNLINEARREGRESSION])] = "提案手法"

# 実験1.5
zikken_id = 2
already_optimize_ok = True # 最適化を再度行うか
n_data_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
n_data_list = [100, 500, 1000, 1500, 2000, 2500]
n_data_list = [2500]
n_feature_list = [10]
is_x_normal = False # xを正規分布に従って発生させるか
opt_list = [FRANKWOLFE, SIMULATEDANNEALING] # SELECTBESTDATA]
opt_list = [FRANKWOLFE, SIMULATEDANNEALING, MATHMATICALOPTIMIZATION, SELECTBESTDATA]
opt_list = [FRANKWOLFE]
# opt_list = [SIMULATEDANNEALING]
# opt_list = [MATHMATICALOPTIMIZATION]
# opt_list = [SELECTBESTDATA]
data_type_list = [LOGX]
g.coef = 10
# data_type_list = [LOGX2]
min_xs = 0
max_xs = 10
problem_list = [INEQUALITYCONSTRAINT]
# n_nearest_list = list(range(10, 1010, 10))
n_nearest_best_rate = 0.075
auto_n_nearest = True
plot_name_dic[(solvers_names[FRANKWOLFE], mlmodel_names[KNNLINEARREGRESSION])] = "提案手法"

#実験2
# zikken_id = 2
# already_optimize_ok = False # 最適化を再度行うか
# n_data_list = [1000]
# # n_data_list = [200]
# n_feature_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# # n_feature_list = [10]
# is_x_normal = True # xを正規分布に従って発生させるか
# x_scale = 3
# opt_list = [MATHMATICALOPTIMIZATION, SIMULATEDANNEALING, FRANKWOLFE, SELECTBESTDATA]
# opt_list = [MATHMATICALOPTIMIZATION, FRANKWOLFE, FRANKWOLFE2, SELECTBESTDATA]
# data_type_list = [SINX0MOUNT2]
# data_type_list = [ACKELY]
# data_type_list = [RASTRIGIN]
# # data_type_list = [XSQUARE]
# auto_n_nearest = True


#Directory
home_dir = "./"
# ml_dir = home_dir + "machine_learning/"
if is_x_normal:
   result_dir = home_dir + f"result/normalize/x_s{x_scale}/"
else:
   result_dir = home_dir + "result/uniform/"

process_dir = result_dir + "process/"
ml_model_dir = result_dir + "machine_learning/"
saved_model_dir = ml_model_dir + "saved_model"
saved_data_dir = result_dir + "data"
ml_shape_dir = ml_model_dir + "shape"
ml_info_dir = ml_model_dir + "info"
opt_dir = result_dir + "optimization"

""" ML手法 オート設定モード"""
constr_ml_for_opt = {SIMULATEDANNEALING: [NEURALNETWORK, LIGHTGBM], TRUSTREGION: [LIGHTGBM], FRANKWOLFE: [WEIGHTEDLINEARREGRESSION, ANNLINEARREGRESSION, KNNLINEARREGRESSION],
                     MATHMATICALOPTIMIZATION: [LINEARREGRESSION, POLYNOMIALREGRESSION, SVRLINEAR, SVRPOLY, SVRGAUSS, CGA2M], STEPDISTANCE: [ANNLINEARREGRESSION],
                     BAYESIANOPTIMIZATIONMU: [GAUSSIANPROCESSREGRESSION], BAYESIANOPTIMIZATIONLCB: [GAUSSIANPROCESSREGRESSION], SELECTBESTDATA: [KNNLINEARREGRESSION],
                     FRANKWOLFE2: [KNNLINEARREGRESSION]}

""" 問題+データタイプ オート設定モード"""
constr_problem_data_for_opt = {NONCONSTRAINT: [SINX0, SINX0MOUNT2, SINX01MOUNT2, COMPLEX7, ROSENBROCK, ACKELY, XSQUARE, MOUNT2, RASTRIGIN, LOGX, LOGX2], 
                               BOXCONSTRAINT: [SINX0, SINX0MOUNT2, SINX01MOUNT2, COMPLEX7, ROSENBROCK, ACKELY, XSQUARE, MOUNT2],
                               BOXFEATURECONSTRAINT: [SINX0, SINX0MOUNT2, SINX01MOUNT2, COMPLEX7, ROSENBROCK, ACKELY, XSQUARE, MOUNT2], 
                               DISKCONSTRAINT: [SINX0, SINX0MOUNT2, SINX01MOUNT2, COMPLEX7, ROSENBROCK, ACKELY, XSQUARE, MOUNT2],
                               INEQUALITYCONSTRAINT: [SINX0, SINX0MOUNT2, SINX01MOUNT2, COMPLEX7, ROSENBROCK, ACKELY, XSQUARE, MOUNT2, LOGX, LOGX2]
                            }

""" 問題+特徴量数　オート設定モード"""
bad_constr_problem_ufeature_for_opt = {NONCONSTRAINT: [], BOXCONSTRAINT: [], BOXFEATURECONSTRAINT: [1], DISKCONSTRAINT: [], INEQUALITYCONSTRAINT: []}

""" データタイプ+特徴量数　オート設定モード"""
bad_constr_data_feature_for_opt = {SINX0: [], SINX0MOUNT2: [], SINX01MOUNT2: [], COMPLEX7: list(range(1, 7)), ROSENBROCK: [1], ACKELY: [1], XSQUARE: [], MOUNT2: [], RASTRIGIN: [], LOGX: [], LOGX2: []}

""" データタイプ+特徴量数　オート設定モード"""
appropriate_data_feature_for_opt = {SINX0: [2], SINX0MOUNT2: [2], SINX01MOUNT2: [2], COMPLEX7: [7], ROSENBROCK: [2], ACKELY: [2], XSQUARE: [2], MOUNT2:[2]}

""" 数理計画法の定式化"""
formalization_ml_problem = {LINEARREGRESSION: LP, WEIGHTEDLINEARREGRESSION: LP, RANDOMFOREST: MIP, LIGHTGBM: MIP, SVRLINEAR: LP, SVRPOLY: NLP, SVRGAUSS: NLP,
                            ANNLINEARREGRESSION: LP, KNNLINEARREGRESSION: LP, CGA2M: MIP, POLYNOMIALREGRESSION: NLP}


if auto_n_nearest:
   n_nearest_list = [int(n_data_list[i] * n_nearest_best_rate) for i in range(len(n_data_list))]