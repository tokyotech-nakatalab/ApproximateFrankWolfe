from utility.setting import *
from utility.tool import *
from utility.module import *
from utility.line_notify import *
from experiment.generate_data import *
from machine_learning.model.cga2m_regression import *
from solvers.ml_modelize.linear_regression import *
from problem.base_problem import *
from machine_learning.tools.select_model import *
from utility.result_manager import *
from utility.experiment_tool import *
import utility.gloabl_values as g
from utility.data_manager import *
from utility.maximize2minimize import *


def main():
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
                                                np.random.seed(g.seed)
                                                set_unique_parameter()
                                                generate_result_save_directory()
                                                if not is_only_shape_check and not already_optimize_ok and check_result_exist():
                                                    continue

                                                print_start()
                                                solver = select_opt_problem()
                                                solver.set_problem()

                                                XS_train, C_train, XS_test, C_test, S_optval, X_init = [], [], [], [], [], []
                                                set_data(solver, XS_train, C_train, XS_test, C_test)
                                                set_val_data(solver, S_optval)
                                                set_init_x(solver, X_init, XS_train, C_train)
                                                fs, learing_time = create_model(XS_train, C_train,  XS_test, C_test)
                                                train_rmse, test_rmse = eval_all_f(fs, XS_train, C_train, XS_test, C_test)
                                                if is_only_shape_check:
                                                    visualize_regression_shape(fs, solver.problem, S_optval, XS_train, C_train)
                                                    continue


                                                if not is_only_learning:
                                                    opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list = [], [], [], [], [], []
                                                    opt_time_list = []
                                                    for g.val in val_list:
                                                        init_x = X_init[g.val]
                                                        np.random.seed(1000*g.seed + g.val)
                                                        tic()
                                                        best_x, best_obj, best_true_obj, ave_rho, ave_diameter, ave_rdnear = solver.optimize(fs, S_optval[g.val], init_x)
                                                        # best_obj, best_true_obj = min2plus(best_obj), min2plus(best_true_obj)
                                                        opt_time_list.append(toc())
                                                        print(f"合計時間：{opt_time_list[-1]}, サーチとフィット：{g.search_time/opt_time_list[-1]}%, 最適化：{g.solve_time/opt_time_list[-1]}")
                                                        opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list = add_result(opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list, best_x, S_optval[g.val], best_obj, best_true_obj, ave_rho, ave_diameter, ave_rdnear)
                                                    print_output(X_init, opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list)
                                                    output_result(X_init, opt_xs_list, obj_list, true_obj_list, ave_rho_list, ave_diameter_list, ave_rdnear_list, opt_time_list, train_rmse, test_rmse, learing_time)
    if not is_only_learning and not is_only_shape_check:
        read_result()
        save_all_pattern_result()
    send_line_notify("実験が終わりました")


def create_model(XS_train, C_train, XS_test, C_test):
    print("学習開始")
    fs, learning_time = create_base_model(XS_train, C_train, XS_test, C_test)
    print("学習終了")
    return fs, learning_time

def create_base_model(XS_train, C_train,  XS_test, C_test):
    fs, time = [], []
    for i in range(g.n_item):
        tic()
        f = select_model(XS_train[i], C_train[i], i)
        f = model_fit(f, XS_train[i], C_train[i], i)
        learning_time = toc()
        f.set_parameter()
        fs.append(f)

        train_rmse, test_rmse = f.eval_performance(XS_train[i], C_train[i], XS_test[i], C_test[i])
        saved_time = save_or_read_ml_info(train_rmse, test_rmse, learning_time, i+1)
        if saved_time == -1:
            time.append(learning_time)
        else:
            time.append(saved_time)
    return fs, sum(time)

def create_good_obj_model(fs, XS_train, C_train, problem):
    for i in range(g.n_item):
        if not fs[i].is_saved_flg:
            fs[i].good_obj_mdl = create_fit_llr(i, XS_train[i], C_train[i], problem.max_bounds[i][0], problem.min_bounds[i][0])
            save_model(fs[i], i)
    return fs

def visualize_regression_shape(fs, problem, S_optval, XS_train, C_train):
    is_3d = g.n_feature != 1
    if g.select_ml == WEIGHTEDLINEARREGRESSION:
        check_regression_shape(fs[0].good_obj_mdl, problem, S_optval[0], XS_train[0], C_train[0], is_3d=is_3d)
    elif g.select_ml == ANNLINEARREGRESSION or g.select_ml == KNNLINEARREGRESSION:
        check_regression_shape(fs[0], problem, S_optval[0], XS_train[0], C_train[0], is_3d=is_3d)
    else:
        check_regression_shape(fs[0], problem, S_optval[0], XS_train[0], C_train[0], is_3d=is_3d)



def experiment_real_data():
    data = pd.read_csv('./experiment/real_data/advertising.csv').values
    x = data[:, 0:3]
    y = data[:, 3]
    kouho_list = list(range(10, 210, 10))

    g.n_item, g.select_opt, g.select_problem, g.select_data_type, g.n_feature, g.n_user_available_x, g.select_ml = 1, MATHMATICALOPTIMIZATION, REALCONSTRAINT, REAL, 3, 3, LINEARREGRESSION
    result_list = []
    for g.n_nearest in [10]: #list(range(10, 210, 10)):
        f = MyLinearRegression(0)
        f.fit(x, y)
        f.set_parameter()
        fs = [f]
        solver = select_opt_problem()
        solver.set_problem()
        init_x = np.array([[0, 0, 0]])
        best_x, best_obj, best_true_obj, ave_rho, ave_diameter = solver.optimize(fs,np.array([[]]), init_x)
        print(best_x, best_obj, best_true_obj, ave_rho, ave_diameter)
        result_list.append([best_x, best_obj, best_true_obj, ave_rho, ave_diameter])
    print("***************************")
    print(result_list)


    # g.n_item, g.select_opt, g.select_problem, g.select_data_type, g.n_feature, g.n_user_available_x, g.select_ml = 1, FRANKWOLFE, REALCONSTRAINT, REAL, 3, 3, KNNLINEARREGRESSION
    # result_list = []
    # for g.n_nearest in kouho_list:
    #     f = MyKnnLinearRegression(0)
    #     f.set_data(x, y)
    #     fs = [f]
    #     solver = select_opt_problem()
    #     solver.set_problem()
    #     init_x = np.array([[0, 0, 0]])
    #     best_x, best_obj, best_true_obj, ave_rho, ave_diameter = solver.optimize(fs,np.array([[]]), init_x)
    #     print(best_x, best_obj, best_true_obj, ave_rho, ave_diameter)
    #     result_list.append([best_x, best_obj, best_true_obj, ave_rho, ave_diameter])
    # x_list = [result_list[i][0][0] for i in range(len(result_list))]
    # df = pd.DataFrame(x_list)
    # df.to_csv("./resulttt.csv")
    # print("***************************")
    # print(result_list)



if __name__ == "__main__":
    if not use_real_data:
        generate_all_data()
        main()
    else:
        experiment_real_data()