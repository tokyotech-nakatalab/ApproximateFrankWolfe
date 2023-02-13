from utility.module import *
from utility.setting import *
import utility.gloabl_values as g

OBJ = 1
DO_TIME = 4
N_DIM = 6
N_DATA = 5

def plot_proposed_method(result, conditions):
    n_seed_dir_str = len(g.result_opt_dir.split("/")[-1])
    base_file = g.result_opt_dir[:-n_seed_dir_str]
    n_pattern = len(seed_list) * len(val_list)
    n_dim = len(n_feature_list)
    x_columns = n_feature_list

    # 目的関数値のプロット
    ave, ave_list = .0, []
    cnt_dim = 0
    for i in range(len(result)):
        ave += result[i][OBJ]
        if i % n_pattern == n_pattern - 1:
            ave /= n_pattern
            ave_list.append(ave)
            ave = .0
            cnt_dim += 1
            if cnt_dim % n_dim == 0:
                name = conditions[i][N_DATA]
                plt.plot(x_columns, ave_list, label=name, marker="o")
                ave_list = []

    file_name = base_file + "obj.png"
    plt.rcParams["font.size"] = 15
    plt.ylabel("f(x)")
    plt.xlabel("n_dim")
    plt.legend()
    plt.xticks(x_columns)
    plt.savefig(file_name)
    plt.close()

    # 実行時間のプロット
    ave, ave_list = .0, []
    cnt_dim = 0
    for i in range(len(result)):
        ave += result[i][DO_TIME]
        if i % n_pattern == n_pattern - 1:
            ave /= n_pattern
            ave_list.append(ave)
            ave = .0
            cnt_dim += 1
            if cnt_dim % n_dim == 0:
                name = conditions[i][N_DATA]
                plt.plot(x_columns, ave_list, label=name, marker="o")
                ave_list = []
    file_name = base_file + "time.png"
    plt.rcParams["font.size"] = 12
    plt.ylabel("time")
    plt.xlabel("n_dim")
    plt.legend()
    plt.xticks(x_columns)
    plt.savefig(file_name)
    plt.close()