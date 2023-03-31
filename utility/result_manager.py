from utility.setting import *
from utility.module import *
from utility.tool import *
from utility.experiment_tool import *
import utility.gloabl_values as g
from utility.plot_propsed import plot_proposed_method


def save_all_pattern_result():
    target_names = {"item": n_item_list, "data": n_data_list, "feature": n_feature_list, "u_feature": n_user_available_x_list, "noise_sigma": noise_sigma_list, "n_nearest": n_nearest_list, "rmse": noise_sigma_list}
    for g.select_problem in problem_list:
        for g.select_data_type in data_type_list:
            if not g.select_data_type in constr_problem_data_for_opt[g.select_problem]:
                continue
            for g.target in target_names.keys():
                set_unique_parameter()
                generate_compare_directory()
                if g.target == "item":
                    conditions = []
                    result = []
                    for g.n_data in n_data_list:
                        for g.n_feature in n_feature_list:
                            for g.n_user_available_x in n_user_available_x_list:
                                for g.n_nearest in n_nearest_list:
                                    for g.noise_sigma in noise_sigma_list:
                                        result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, n_compare_plot_list = {}, {}, {}, {}, {}, {}
                                        for g.seed in seed_list:
                                            for g.val in val_list:
                                                g.seed_val = g.seed * len(val_list) + g.val
                                                for g.n_item in n_item_list:
                                                    for g.select_opt in opt_list:
                                                        for g.select_ml in ml_list:
                                                            if not check_constraint():
                                                                continue
                                                            else:
                                                                n_compare_plot_list[g.n_item] = True
                                                            key = get_dic_key()
                                                            r = g.result_dic[key]
                                                            result.append([r["obj"], r["true_obj"], r["opt_time"], r["learning_time"], r["total_time"], r["average_rho"], r["average_diameter"], r["average_rdnear"]])
                                                            conditions.append(generate_condition_list())
                                                            add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot,r)
                                        plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot)
                    if len(result) != 0:
                        set_unique_parameter()
                        columns_names = ["シード", "検証数", "最適化手法", "機械学習手法", "アイテム数", "データ数", "特徴量数", "ユーザ可変数", "ノイズ", "近傍"] + ["目的関数値", "真の目的関数値", "最適化時間", "学習時間", "総時間", "平均最小特異値", "M近傍の平均直径", "RDM"]
                        file_name = g.compare_dir + "/all_result.csv"
                        result_data = np.concatenate([conditions, result], axis=1)
                        pd.DataFrame(result_data, columns=columns_names).to_csv(file_name, index=False)
                        if is_plot_proposed_method:
                            plot_proposed_method(result, conditions)
                if g.target == "data":
                    for g.n_item in n_item_list:
                        for g.n_feature in n_feature_list:
                            # for g.n_nearest in n_nearest_list:
                                for g.n_user_available_x in n_user_available_x_list:
                                    for g.noise_sigma in noise_sigma_list:
                                        result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, n_compare_plot_list = {}, {}, {}, {}, {}, {}
                                        for g.seed in seed_list:
                                            for g.val in val_list:
                                                g.seed_val = g.seed * len(val_list) + g.val
                                                for g.n_data in n_data_list:
                                                    g.n_nearest = int(g.n_data * n_nearest_best_rate)
                                                    for g.select_opt in opt_list:
                                                        for g.select_ml in ml_list:
                                                            if not check_constraint():
                                                                continue
                                                            else:
                                                                n_compare_plot_list[g.n_data] = True
                                                            key = get_dic_key()
                                                            r = g.result_dic[key]
                                                            add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot,r)
                                        plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot)
                if g.target == "feature":
                    for g.n_item in n_item_list:
                        for g.n_data in n_data_list:
                            for g.n_nearest in n_nearest_list:
                                for g.n_user_available_x in n_user_available_x_list:
                                    for g.noise_sigma in noise_sigma_list:
                                        result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, n_compare_plot_list = {}, {}, {}, {}, {}, {}
                                        for g.seed in seed_list:
                                            for g.val in val_list:
                                                g.seed_val = g.seed * len(val_list) + g.val
                                                for g.n_feature in n_feature_list:
                                                    # 特別な処理
                                                    g.n_user_available_x = g.n_feature
                                                    for g.select_opt in opt_list:
                                                        for g.select_ml in ml_list:
                                                            if not check_constraint():
                                                                continue
                                                            else:
                                                                n_compare_plot_list[g.n_feature] = True
                                                            key = get_dic_key()
                                                            r = g.result_dic[key]
                                                            add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot,r)
                                        plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot)
                # if g.target == "u_feature":
                #     for g.n_item in n_item_list:
                #         for g.n_data in n_data_list:
                #             for g.n_feature in n_feature_list:
                #                 for g.noise_sigma in noise_sigma_list:
                #                     result_dic_for_plot, time_dic_for_plot, n_compare_plot_list = {}, {}, {}
                #                     for g.seed in seed_list:
                #                         for g.val in val_list:
                #                             g.seed_val = g.seed * len(val_list) + g.val
                #                             for g.n_user_available_x in n_user_available_x_list:
                #                                 for g.select_opt in opt_list:
                #                                     for g.select_ml in ml_list:
                #                                         if not check_constraint():
                #                                             continue
                #                                         else:
                #                                             n_compare_plot_list[g.n_feature] = True
                #                                         key = get_dic_key()
                #                                         r = g.result_dic[key]
                #                                         add_result_for_plot(result_dic_for_plot, time_dic_for_plot, r)
                #                     plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot)
                if g.target == "noise_sigma":
                    for g.n_item in n_item_list:
                        for g.n_data in n_data_list:
                            for g.n_nearest in n_nearest_list:
                                for g.n_feature in n_feature_list:
                                    for g.n_user_available_x in n_user_available_x_list:
                                        result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, n_compare_plot_list = {}, {}, {}, {}, {}, {}
                                        for g.seed in seed_list:
                                            for g.val in val_list:
                                                g.seed_val = g.seed * len(val_list) + g.val
                                                for g.noise_sigma in noise_sigma_list:
                                                    for g.select_opt in opt_list:
                                                        for g.select_ml in ml_list:
                                                            if not check_constraint():
                                                                continue
                                                            else:
                                                                n_compare_plot_list[get_noise()[g.noise_sigma]] = True
                                                            key = get_dic_key()
                                                            r = g.result_dic[key]
                                                            add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot,r)
                                        plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot)
                if g.target == "n_nearest":
                    for g.n_item in n_item_list:
                        for g.n_data in n_data_list:
                            for g.noise_sigma in noise_sigma_list:
                                for g.n_feature in n_feature_list:
                                    for g.n_user_available_x in n_user_available_x_list:
                                        result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, n_compare_plot_list = {}, {}, {}, {}, {}, {}
                                        for g.seed in seed_list:
                                            for g.val in val_list:
                                                g.seed_val = g.seed * len(val_list) + g.val
                                                for g.n_nearest in n_nearest_list:
                                                    for g.select_opt in opt_list:
                                                        for g.select_ml in ml_list:
                                                            if not check_constraint():
                                                                continue
                                                            else:
                                                                n_compare_plot_list[g.n_nearest] = True
                                                            key = get_dic_key()
                                                            r = g.result_dic[key]
                                                            add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot,r)
                                        plot_result(n_compare_plot_list.keys(), result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot)
                # if g.target == "rmse":
                #     for g.n_item in n_item_list:
                #         for g.n_data in n_data_list:
                #             for g.n_feature in n_feature_list:
                #                 for g.n_user_available_x in n_user_available_x_list:
                #                     result_dic_for_plot, rmse_dic_for_plot, n_compare_plot_list = {}, {}, {}
                #                     for g.seed in seed_list:
                #                         for g.val in val_list:
                #                             g.seed_val = g.seed * len(val_list) + g.val
                #                             for g.noise_sigma in noise_sigma_list:
                #                                 for g.select_opt in opt_list:
                #                                     for g.select_ml in ml_list:
                #                                         if not check_constraint():
                #                                             continue
                #                                         else:
                #                                             n_compare_plot_list[g.noise_sigma] = True
                #                                         key = get_dic_key()
                #                                         r = g.result_rmse_dic[key]
                #                                         add_result_rmse_for_plot(result_dic_for_plot, rmse_dic_for_plot, r)
                #                     plot_result_rmse(result_dic_for_plot, rmse_dic_for_plot)


def plot_result(compare_list, result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot):
    set_unique_parameter()
    n_yokoziku = len(compare_list)
    fontsize_label = 24
    fontsize_hanrei = 20
    fontsize_memori = 18
    
    if len(compare_list) != 1 and len(result_dic_for_plot) > 0:
        fig, ax1 = plt.subplots(1,1,figsize=(10,8))
        x_zahyo = [i for i in range(len(compare_list))]

        all_ave_list = []
        index_list = []
        for key in result_dic_for_plot.keys():
            objs = np.array(result_dic_for_plot[key])
            present_id = list(range(objs.shape[1]))
            if g.select_problem == INEQUALITYCONSTRAINT:
                for i in range(objs.shape[1]):
                    for j in range(objs.shape[0]):
                        if objs[j, i] == 10 ** -5:
                            present_id.remove(i)
                            break
            objs = -1 * objs
            ave_obj, std_obj = np.nanmean(objs, axis=0), np.nanstd(objs, axis=0)
            if zikken_id == 1:
                x_zahyo_adjust = np.array(n_nearest_list)
            else:
                x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
            ax1.errorbar(x_zahyo_adjust[present_id], ave_obj[present_id], yerr = std_obj[present_id], capsize=5, fmt='o', markersize=4, label=plot_name_dic[key], color=color_dic[key])
            all_ave_list.append(list(ave_obj))
            index_list.append(plot_name_dic[key])
        pd.DataFrame(all_ave_list, index=index_list, columns=[f"{c}" for c in compare_list]).to_csv(g.compare_target_dir + f"/obj_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}_all_ave.csv")
        if n_yokoziku > 10:
            ax1.tick_params(labelsize=fontsize_memori)
        else:
            ax1.set_xticks(x_zahyo)
            ax1.set_xticklabels([f"{c}" for c in compare_list])
        ax1.tick_params(labelsize=fontsize_memori)
        if is_taisu:
            ax1.set_yscale('log')
        if g.target == "n_nearest":
            ax1.set_xlabel("$M$", fontsize=fontsize_label, fontweight='bold')
        elif g.target == "feature":
            ax1.set_xlabel("$n$", fontsize=fontsize_label, fontweight='bold')
        elif g.target == "data":
            ax1.set_xlabel("$N$", fontsize=fontsize_label, fontweight='bold')
        else:
            ax1.set_xlabel(g.target, fontsize=fontsize_label, fontweight='bold')
        ax1.set_ylabel("目的関数値", fontname='MS Gothic', fontsize=fontsize_label, fontweight='bold')
        ax1.grid(True)
        ax1.legend(prop={'family':'Yu Gothic', "size":fontsize_hanrei, "weight":"bold"})
        file_name = g.compare_target_dir + f"/obj_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
        fig.savefig(file_name)
        plt.close()

        # time 
        ######################################################################################
        fig, ax1 = plt.subplots(1,1,figsize=(10,8))
        x_zahyo = [i for i in range(len(compare_list))]

        for key in time_dic_for_plot.keys():
            times = np.array(time_dic_for_plot[key])
            present_id = list(range(objs.shape[1]))
            if g.select_problem == INEQUALITYCONSTRAINT:
                for i in range(times.shape[1]):
                    for j in range(times.shape[0]):
                        if times[j, i] == 10 ** -5:
                            present_id.remove(i)
                            break
            ave_time, std_time = np.nanmean(times, axis=0), np.nanstd(times, axis=0)
            if zikken_id == 1:
                x_zahyo_adjust = np.array(n_nearest_list)
            else:
                x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
            ax1.errorbar(x_zahyo_adjust[present_id], ave_time[present_id], yerr = std_time[present_id], capsize=5, fmt='o', markersize=4, label=plot_name_dic[key], color=color_dic[key])

        # ax1.set_ylim(-0.1, 2.0)
        if n_yokoziku > 10:
            ax1.tick_params(labelsize=fontsize_memori)
        else:
            ax1.set_xticks(x_zahyo, fontsize=fontsize_memori)
            ax1.set_xticklabels([f"{c}" for c in compare_list])
        ax1.tick_params(labelsize=fontsize_memori)
        if is_taisu:
            ax1.set_yscale('log')
        if g.target == "n_nearest":
            ax1.set_xlabel("$M$", fontsize=fontsize_label, fontweight='bold')
        elif g.target == "feature":
            ax1.set_xlabel("$n$", fontsize=fontsize_label, fontweight='bold')
        elif g.target == "data":
            ax1.set_xlabel("$N$", fontsize=fontsize_label, fontweight='bold')
        else:
            ax1.set_xlabel(g.target, fontsize=fontsize_label, fontweight='bold')
        ax1.set_ylabel("実行時間 (秒)", fontname='MS Gothic', fontsize=fontsize_label, fontweight='bold')
        # ax1.grid(True)
        ax1.legend(prop={'family':'Yu Gothic', "size":fontsize_hanrei, "weight":"bold"})
        file_name = g.compare_target_dir + f"/time_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
        fig.savefig(file_name)
        plt.close()

        # rho
        ######################################################################################
        if zikken_id == 1:
            proposed_key = (solvers_names[FRANKWOLFE2], mlmodel_names[KNNLINEARREGRESSION])

            fig, ax1 = plt.subplots(1,1,figsize=(10,8))
            x_zahyo = [i for i in range(len(compare_list))]

            rhos = np.array(rho_dic_for_plot[proposed_key])
            ave_rho, std_rho = np.average(rhos, axis=0), np.std(objs, axis=0)
            # x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[proposed_key]
            x_zahyo_list = n_nearest_list
            if use_errorbar:
                ax1.errorbar(x_zahyo_list, ave_rho, yerr = std_rho, capsize=5, fmt='o', markersize=4, color=color_dic[proposed_key])
            else:
                ax1.scatter(x_zahyo_list, ave_rho, color=color_dic[proposed_key])
            if n_yokoziku > 10:
                ax1.tick_params(labelsize=fontsize_memori)
            else:
                ax1.set_xticks(x_zahyo)
                ax1.set_xticklabels([f"{c}" for c in compare_list])
            ax1.tick_params(labelsize=fontsize_memori)
            # if is_taisu:
            ax1.set_yscale('log')
            if g.target == "n_nearest":
                ax1.set_xlabel("$M$", fontsize=fontsize_label, fontweight='bold')
            else:
                ax1.set_xlabel(g.target, fontsize=fontsize_label, fontweight='bold')
            ax1.set_ylabel(r"$\rho^{-1}_{j,\min}$の平均値", fontname='MS Gothic', fontsize=fontsize_label, fontweight='bold')
            ax1.grid(True)
            # ax1.legend(fontsize=fontsize)
            file_name = g.compare_target_dir + f"/rho_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
            fig.savefig(file_name)
            plt.close()

            # diam
            #####################################################################################

            fig, ax1 = plt.subplots(1,1,figsize=(10,8))
            diameters = np.array(diameter_dic_for_plot[proposed_key])
            ave_diameter, std_diameter = np.average(diameters, axis=0), np.std(objs, axis=0)
            if use_errorbar:
                ax1.errorbar(x_zahyo_list, ave_diameter, yerr = std_diameter, capsize=5, fmt='o', markersize=4, color=color_dic[proposed_key])
            else:
                ax1.scatter(x_zahyo_list, ave_diameter, color=color_dic[proposed_key])
            if n_yokoziku > 10:
                ax1.tick_params(labelsize=fontsize_memori)
            else:
                ax1.set_xticks(x_zahyo)
                ax1.set_xticklabels([f"{c}" for c in compare_list])
            ax1.tick_params(labelsize=fontsize_memori)
            # if is_taisu:
            ax1.set_yscale('log')
            if g.target == "n_nearest":
                ax1.set_xlabel("$M$", fontsize=fontsize_label, fontweight='bold')
            else:
                ax1.set_xlabel(g.target, fontsize=fontsize_label, fontweight='bold')
            ax1.set_ylabel(r"$D_j$ の平均値", fontname='MS Gothic', fontsize=fontsize_label, fontweight='bold')
            ax1.grid(True)
            # ax1.legend(fontsize=18)
            file_name = g.compare_target_dir + f"/diam_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
            fig.savefig(file_name)
            plt.close()

            # rdm
            #####################################################################################

            fig, ax1 = plt.subplots(1,1,figsize=(10,8))
            rdnears = np.array(rdnear_dic_for_plot[proposed_key])
            ave_rdnear, std_rdnear = np.average(rdnears, axis=0), np.std(objs, axis=0)
            # print(np.nanargmin(ave_rdnear))
            # print(ave_rdnear)
            if not use_errorbar:
                # print(np.isnan(ave_rdnear))
                not_nan_idx = [i for i in range(ave_rdnear.size) if not np.isnan(ave_rdnear[i])]
                ax1.errorbar(np.array(x_zahyo_list)[not_nan_idx], ave_rdnear[not_nan_idx], yerr = std_rdnear[not_nan_idx], capsize=5, fmt='o', markersize=4, color=color_dic[proposed_key])
            else:
                ax1.scatter(x_zahyo_list, ave_rdnear, color=color_dic[proposed_key])
            if n_yokoziku > 10:
                ax1.tick_params(labelsize=fontsize_memori)
            else:
                ax1.set_xticks(x_zahyo)
                ax1.set_xticklabels([f"{c}" for c in compare_list])
            ax1.tick_params(labelsize=fontsize_memori)
            # if is_taisu:
            ax1.set_yscale('log')
            if g.target == "n_nearest":
                ax1.set_xlabel("$M$", fontsize=fontsize_label, fontweight='bold')
            else:
                ax1.set_xlabel(g.target, fontsize=fontsize_label, fontweight='bold')
            ax1.set_ylabel(r"$\sqrt{M} \frac{D_j^2}{\rho_{j,\min}}$ の平均値", fontname='MS Gothic', fontsize=fontsize_label-2, fontweight='bold')
            ax1.grid(True)
            # ax1.legend(fontsize=18)
            file_name = g.compare_target_dir + f"/rdm_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
            fig.savefig(file_name)
            plt.close()
            

# def plot_result(compare_list, result_dic_for_plot, time_dic_for_plot):
#     set_unique_parameter()
#     if len(compare_list) != 1 and len(result_dic_for_plot) > 0:
#         for n_plt in ["all"]:
#             for plt_select in ["scatter", "bar"]:
#                 fig, ax1 = plt.subplots(1,1,figsize=(10,8))
#                 x_zahyo = [i for i in range(len(compare_list))]
#                 if plt_select == "bar":
#                     ax2 = ax1.twinx()
#                     ax2.set_ylabel("実行時間", fontname='MS Gothic')

#                 for key in result_dic_for_plot.keys():
#                     if n_plt == "noline" and key == (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[LINEARREGRESSION]):
#                         continue
#                     res = np.array(result_dic_for_plot[key])
#                     times = np.array(time_dic_for_plot[key])
#                     if plt_select == "scatter":
#                         x_zahyo_adjust = np.tile(x_zahyo, res.shape[0]) + plot_coordinate_dic[key]
#                         res = res.reshape(1, -1)
#                         ax1.scatter(x_zahyo_adjust, res, label=key, color=color_dic[key])
#                     elif plt_select == "bar":
#                         ave_acc = np.average(res, axis=0)
#                         std_acc = np.std(res, axis=0)
#                         x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
#                         ax1.errorbar(x_zahyo_adjust, ave_acc, yerr = std_acc, capsize=5, fmt='o', markersize=4, label=key, color=color_dic[key])

#                         ave_time = np.average(times, axis=0)
#                         std_time = np.std(times, axis=0)
#                         ax2.plot(ave_time, label=key, color=color_dic[key])
#                 ax1.set_xticks(x_zahyo)
#                 ax1.set_xticklabels([f"{c}" for c in compare_list])
#                 ax1.set_xlabel(g.target)
#                 ax1.set_ylabel("真の目的関数値", fontname='MS Gothic')

#                 ax1.grid(True)
#                 ax1.legend()
#                 if plt_select == "scatter":
#                     file_name = g.compare_target_dir + f"/sca_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}_{n_plt}" + ".png"
#                 elif plt_select == "bar":
#                     file_name = g.compare_target_dir + f"/bar_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}_{n_plt}" + ".png"
#                 fig.savefig(file_name)
#                 plt.close()

#                 if plt_select == "bar":
#                     plt.figure(figsize=(6,4))
#                     for key in result_dic_for_plot.keys():
#                         res = np.array(result_dic_for_plot[key])
#                         ave_acc = np.average(res, axis=0)
#                         std_acc = np.std(res, axis=0)
#                         x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
#                         plt.plot(x_zahyo, ave_acc, marker="o", label=plot_name_dic[key])
#                         # plt.errorbar(x_zahyo_adjust, ave_acc, yerr = std_acc, capsize=5, fmt='o', markersize=4, label=key, color=color_dic[key])
#                     file_name = g.compare_target_dir + f"/s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}_acc" + ".png"
#                     plt.ylabel("f(x)")
#                     plt.xlabel("n_dim")
#                     plt.legend()
#                     plt.xticks(x_zahyo, [f"{c}" for c in compare_list])
#                     plt.rcParams["font.size"] = 15
#                     plt.savefig(file_name)
#                     plt.close()

#                     plt.figure(figsize=(6,4))
#                     for key in result_dic_for_plot.keys():
#                         times = np.array(time_dic_for_plot[key])
#                         ave_time = np.average(times, axis=0)
#                         std_time = np.std(times, axis=0)
#                         x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
#                         plt.plot(x_zahyo, ave_time, marker="o", label=plot_name_dic[key])
#                         # plt.errorbar(x_zahyo, ave_time, yerr = std_time, capsize=5, fmt='o', markersize=4, label=key, color=color_dic[key])
#                     file_name = g.compare_target_dir + f"/s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}_time" + ".png"
#                     plt.ylabel("f(x)")
#                     plt.xlabel("n_dim")
#                     plt.legend()
#                     plt.xticks(x_zahyo, [f"{c}" for c in compare_list])
#                     plt.rcParams["font.size"] = 15
#                     plt.savefig(file_name)
#                     plt.close()



# def plot_result(compare_list, result_dic_for_plot, time_dic_for_plot):
#     set_unique_parameter()
#     if len(compare_list) != 1 and len(result_dic_for_plot) > 0:
#         for n_plt in ["all","noline"]:
#             for plt_select in ["scatter", "bar"]:
#                 fig, ax1 = plt.subplots(1,1,figsize=(10,8))
#                 x_zahyo = [i for i in range(len(compare_list))]
#                 if plt_select == "bar":
#                     ax2 = ax1.twinx()
#                     ax2.set_ylabel("実行時間", fontname='MS Gothic')

#                 for key in result_dic_for_plot.keys():
#                     if n_plt == "noline" and key == (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[LINEARREGRESSION]):
#                         continue
#                     res = np.array(result_dic_for_plot[key])
#                     times = np.array(time_dic_for_plot[key])
#                     if plt_select == "scatter":
#                         x_zahyo_adjust = np.tile(x_zahyo, res.shape[0]) + plot_coordinate_dic[key]
#                         res = res.reshape(1, -1)
#                         ax1.scatter(x_zahyo_adjust, res, label=key, color=color_dic[key])
#                     elif plt_select == "bar":
#                         ave = np.average(res, axis=0)
#                         std = np.std(res, axis=0)
#                         x_zahyo_adjust = np.array(x_zahyo) + plot_coordinate_dic[key]
#                         ax1.errorbar(x_zahyo_adjust, ave, yerr = std, capsize=5, fmt='o', markersize=4, label=key, color=color_dic[key])

#                         ave = np.average(times, axis=0)
#                         std = np.std(times, axis=0)
#                         ax2.plot(ave, label=key, color=color_dic[key])
#                 ax1.set_xticks(x_zahyo)
#                 ax1.set_xticklabels([f"{c}" for c in compare_list])
#                 ax1.set_xlabel(g.target)
#                 ax1.set_ylabel("真の目的関数値", fontname='MS Gothic')

#                 ax1.grid(True)
#                 ax1.legend()
#                 if plt_select == "scatter":
#                     file_name = g.compare_target_dir + f"/sca_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_{n_plt}" + ".png"
#                 elif plt_select == "bar":
#                     file_name = g.compare_target_dir + f"/bar_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_{n_plt}" + ".png"
#                 fig.savefig(file_name)
#                 plt.close()

def plot_result_rmse(result_dic_for_plot, rmse_dic_for_plot):
    set_unique_parameter()
    if len(result_dic_for_plot) > 0:
        for key in result_dic_for_plot.keys():
            x_zahyo = np.array(rmse_dic_for_plot[key]).reshape(1, -1)
            res = np.array(result_dic_for_plot[key])
            res = res.reshape(1, -1)
            plt.scatter(x_zahyo, res, label=key, color=color_dic[key])
        plt.legend()
        plt.xlabel(g.target)
        plt.ylabel("真の目的関数値", fontname="MS Gothic")
        file_name = g.compare_target_dir + f"/sca_s{g.seed}_I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}" + ".png"
        plt.savefig(file_name)
        plt.close()


def print_output(init_x_list, opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list):
    print("------------------------------------------------------------------------------------")
    print("")
    print("")
    for i, (init_x, opt_x, obj, true_obj, ave_rho, ave_diameter, ave_rdnear) in enumerate(zip(init_x_list, opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list)):
        print(f"検証データ：{i} 初期解：{init_x}   最終解：{opt_x}   目的関数値：{obj}  真の目的関数値：{true_obj} 平均最小特異値：{ave_rho} M近傍の平均直径: {ave_diameter} RDM: {ave_rdnear}")

def num2str(data):
    for i in val_list:
        for j in range(len(data[i])):
            data[i][j] = str(np.round(data[i][j], 5))
    return data


def get_dic_key():
    return (g.select_problem, g.select_data_type, g.select_opt, g.select_ml, g.n_item, g.n_data, g.n_feature, g.n_user_available_x, g.noise_sigma, g.n_nearest, g.seed, g.val)

def add_result_for_plot(result_dic_for_plot, time_dic_for_plot, rho_dic_for_plot, diameter_dic_for_plot, rdnear_dic_for_plot, r):
    true_obj, time, rho, diameter, rdnear = r["true_obj"], r["total_time"], r["average_rho"], r["average_diameter"], r["average_rdnear"]
    key = get_result_dic_key()
    if not key in result_dic_for_plot.keys():
        result_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
        time_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
        rho_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
        diameter_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
        rdnear_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
    result_dic_for_plot[key][g.seed_val].append(true_obj)
    time_dic_for_plot[key][g.seed_val].append(time)
    rho_dic_for_plot[key][g.seed_val].append(rho)
    diameter_dic_for_plot[key][g.seed_val].append(diameter)
    rdnear_dic_for_plot[key][g.seed_val].append(rdnear)


def add_result_rmse_for_plot(result_dic_for_plot, rmse_dic_for_plot, r):
    true_obj, rmse = r["true_obj"], r["rmse"]
    key = get_result_dic_key()
    if not key in result_dic_for_plot.keys():
        result_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
        rmse_dic_for_plot[key] = [[] for _ in range(len(val_list) * len(seed_list))]
    result_dic_for_plot[key][g.seed_val].append(true_obj)
    rmse_dic_for_plot[key][g.seed_val].append(rmse)
    

def get_result_dic_key():
    return (solvers_names[g.select_opt], mlmodel_names[g.select_ml])

def output_result(init_x_list, opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list, opt_time_list, train_rmse, test_rmse, learing_time):
    data = [[obj_list[i], true_obj_list[i], opt_time_list[i], ave_rho_list[i], ave_diameter_list[i], ave_rdnear_list[i]] + init_x_list[i].ravel().tolist() + opt_xs_list[i] + train_rmse + test_rmse + [learing_time] for i in val_list]
    data = num2str(data)
    f_rmse_feature_name = []
    for i in range(g.n_item):
        f_rmse_feature_name += [f"初期{i}:{g.feature_names[i][j]}" for j in range(g.n_user_available_x)]
    for i in range(g.n_item):
        f_rmse_feature_name += [f"最終{i}:{g.feature_names[i][j]}" for j in range(g.n_feature)]
    for i in range(g.n_item):
        f_rmse_feature_name.extend([f"{i}:tr_rmse", f"{i}:ts_rmse"])
    df = pd.DataFrame(data, columns=["obj", "true_obj", "opt_time", "ave_rho", "ave_diameter", "ave_rdnear"] + f_rmse_feature_name + ["learning_time"])
    file_name = generate_result_file_name()
    df.to_csv(file_name, index=False)


def check_result_exist():
    file_name = generate_result_file_name()
    return os.path.exists(file_name)


def save_result(obj, true_obj, test_rmse):
    key = get_dic_key()
    g.result_dic[key] = [obj, true_obj]
    g.result_rmse_dic[key] = [test_rmse, true_obj]


def add_result(opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list, opt_x, s, obj, true_obj, ave_rho, ave_diameter, ave_rdnear):
    obj_list.append(obj)
    true_obj_list.append(true_obj)
    if g.n_environment_s != 0:
        xs = np.concatenate([np.array(opt_x), np.array(s)], axis=1)
    else:
        xs = np.array(opt_x)
    xs = list(xs.ravel())
    opt_xs_list.append(xs)
    ave_rho_list.append(ave_rho)
    ave_diameter_list.append(ave_diameter)
    ave_rdnear_list.append(ave_rdnear)
    return opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list

def read_result():
    g.result_dic, g.result_rmse_dic = {}, {}
    for g.select_problem in problem_list:
        for g.select_data_type in data_type_list:
            for g.select_opt in opt_list:
                for g.select_ml in ml_list:
                    for g.seed in seed_list:
                        for g.n_item in n_item_list:
                            for g.n_data in n_data_list:
                                for g.n_feature in n_feature_list:
                                    for g.n_user_available_x in n_user_available_x_list:
                                        for g.noise_sigma in noise_sigma_list:
                                            for g.n_nearest in n_nearest_list:
                                                if not check_constraint():
                                                    continue
                                                set_unique_parameter()
                                                generate_result_save_directory()

                                                file_name = generate_result_file_name()
                                                df = pd.read_csv(file_name)
                                                columns = df.columns.tolist()
                                                values = df.values
                                                try:
                                                    obj_i, true_obj_i, rmse_i, opt_time_i, learning_time_i, ave_rho_i, ave_diameter_i, ave_rdnear_i = columns.index("obj"), columns.index("true_obj"), columns.index("0:ts_rmse"), columns.index("opt_time"), columns.index("learning_time"), columns.index("ave_rho"), columns.index("ave_diameter"), columns.index("ave_rdnear")
                                                    obj_list, true_obj_list, test_rmse_list, opt_time_list, learning_time_list, ave_rho_list, ave_diameter_list, ave_rdnear_list  = values[:, obj_i], values[:, true_obj_i], values[:, rmse_i], values[:, opt_time_i], values[:, learning_time_i], values[:, ave_rho_i], values[:, ave_diameter_i], values[:, ave_rdnear_i]
                                                except:
                                                    obj_i, true_obj_i, rmse_i, opt_time_i, learning_time_i, ave_rho_i = columns.index("obj"), columns.index("true_obj"), columns.index("0:ts_rmse"), columns.index("opt_time"), columns.index("learning_time"), columns.index("ave_rho")
                                                    obj_list, true_obj_list, test_rmse_list, opt_time_list, learning_time_list, ave_rho_list, ave_diameter_list  = values[:, obj_i], values[:, true_obj_i], values[:, rmse_i], values[:, opt_time_i], values[:, learning_time_i], values[:, ave_rho_i], values[:, ave_rho_i]                                                    
                                                for g.val in val_list:
                                                    obj, true_obj, test_rmse, opt_time, learning_time, average_rho, average_diameter, average_rdnear = obj_list[g.val], true_obj_list[g.val], test_rmse_list[g.val], opt_time_list[g.val], learning_time_list[g.val], ave_rho_list[g.val], ave_diameter_list[g.val], ave_rdnear_list[g.val]
                                                    # if force_minimize and true_obj < 0:
                                                    #     true_obj = -1 * true_obj
                                                    total_time = opt_time + learning_time
                                                    key = get_dic_key()
                                                    g.result_dic[key] = {"obj": obj, "true_obj": true_obj, "opt_time": opt_time, "learning_time": learning_time, "total_time": total_time, "average_rho": average_rho, "average_diameter": average_diameter, "average_rdnear": average_rdnear}
                                                    g.result_rmse_dic[key] = {"rmse": test_rmse, "true_obj": true_obj}