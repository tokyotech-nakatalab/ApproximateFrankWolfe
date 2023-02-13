from utility.setting import *
from utility.module import *
from experiment.generate_data import *
from utility.result_manager import *
import utility.gloabl_values as g
from utility.tool import print_start

def generate_all_data():
    for g.select_problem in problem_list:
        for g.select_data_type in data_type_list:
            for g.select_opt in opt_list:
                for g.select_ml in ml_list:
                    for g.seed in seed_list:
                        for g.n_data in n_data_list:
                            for g.n_feature in n_feature_list:
                                for g.n_user_available_x in n_user_available_x_list:
                                    for g.noise_sigma in noise_sigma_list:
                                        for g.n_nearest in n_nearest_list:
                                            g.n_item = max(n_item_list)
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
                                            # set_init_x(solver, X_init, XS_train, C_train)