from experiment.generate_data import BaseDataGeneration, Complex7
from utility.setting import *
from utility.module import *
import utility.gloabl_values as g


def set_unique_parameter():
    """ パラメータによって一意に定まるパラメータ """
    g.user_available_x = list(range(g.n_user_available_x))
    g.environment_s = list(set(list(range(g.n_feature))) - set(g.user_available_x))
    g.n_environment_s = len(g.environment_s)
    g.feature_names = {j: [f"x{i}" if i in g.user_available_x else f"s{i}" for i in range(g.n_feature)] for j in range(g.n_item)}
    g.n_train_data = g.n_data * train_rate

    if g.n_nearest > g.n_data:
        g.n_nearest = g.n_data // 4

    """ ディレクトリの名前"""
    g.problem_dir = f"{problem_names[g.select_problem]}"
    g.data_type_dir = f"{data_type_names[g.select_data_type]}"
    g.opt_method_dir = f"{optimization_methods_names[g.select_opt]}"
    g.ml_method_dir = f"{mlmodel_names[g.select_ml]}"
    g.seed_dir = f"seed{g.seed}"

    g.result_file_name = f"I{g.n_item}_d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}"
    g.data_file_name = f"d{g.n_feature}_dx{g.n_user_available_x}_ns{g.noise_sigma}_da{g.n_data}_ne{g.n_nearest}"
    # if g.select_ml == WEIGHTEDLINEARREGRESSION:
    #     g.result_file_name += f"_ws{g.weight_sigma}"

def generate_model_file_name(i):
    model_output = g.result_ml_model_dir + "/" + g.result_file_name + f"_i{i}"
    return  model_output + ".pkl"

def generate_shape_image_name(is_3d=False):
    model_output = g.result_ml_shape_dir + "/" + g.result_file_name
    if is_3d:
        model_output1 = model_output + f"_3d" + ".png"
        model_output2 = model_output + f"_cm" + ".png"
        model_output3 = model_output + f"_data" + ".png"
        return model_output1, model_output2, model_output3
    else:
        return  model_output + ".png"

def generate_process_image_name(iter=0, is_3d=False):
    model_output = g.process_dir + "/" + g.result_file_name
    if is_3d:
        model_output1 = model_output + f"_3d_{iter}" + ".png"
        model_output2 = model_output + f"_cm_{iter}" + ".png"
        model_output3 = model_output + f"_data_{iter}" + ".png"
        return model_output1, model_output2, model_output3
    else:
        return  model_output + ".png"

def generate_gif_name():
    model_output = g.process_dir + "/" + g.result_file_name + "_0all.gif"
    return model_output

def generate_mlinfo_name():
    model_output = g.result_ml_info_dir + "/" + g.result_file_name
    return  model_output + ".csv"

def generate_result_file_name():
    model_output = g.result_opt_dir + "/" + g.result_file_name
    return  model_output + ".csv"


def generate_data_file_name(type, i):
    os.makedirs(g.result_data_dir + "/" + g.data_file_name, exist_ok=True)
    data_output = g.result_data_dir + "/" + g.data_file_name + "/" + f"i{i}" + f"_{type}"
    return  data_output + ".pkl"


def generate_base_directory():
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ml_model_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(ml_shape_dir, exist_ok=True)
    os.makedirs(ml_info_dir, exist_ok=True)
    os.makedirs(saved_data_dir, exist_ok=True)
    os.makedirs(process_dir, exist_ok=True)


def generate_result_save_directory():
    generate_base_directory()
    dir1, dir2, dir3, dir4, dir5, dir6 = opt_dir, saved_model_dir, ml_shape_dir, ml_info_dir, saved_data_dir, process_dir
    for i, dir_name in enumerate([g.problem_dir, g.data_type_dir, g.opt_method_dir, g.ml_method_dir, g.seed_dir]):
        dir1 = dir1 + "/" + dir_name
        dir6 = dir6 + "/" + dir_name
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir6, exist_ok=True)
        if i != 0 and i != 2:
            dir2 = dir2 + "/" + dir_name
            dir3 = dir3 + "/" + dir_name
            dir4 = dir4 + "/" + dir_name
            os.makedirs(dir2, exist_ok=True)
            os.makedirs(dir3, exist_ok=True)
            os.makedirs(dir4, exist_ok=True)
        if i == 1 or i == 4:
            dir5 = dir5 + "/" + dir_name
            os.makedirs(dir5, exist_ok=True)     
    g.result_opt_dir, g.result_ml_model_dir, g.result_ml_shape_dir, g.result_ml_info_dir, g.result_data_dir, g.process_dir = dir1, dir2, dir3, dir4, dir5, dir6


def generate_compare_directory():
    g.compare_dir = opt_dir + "/" + g.problem_dir + "/" + g.data_type_dir + "/compare"
    g.compare_target_dir = g.compare_dir + f"/{g.target}" 
    os.makedirs(g.compare_dir, exist_ok=True)
    os.makedirs(g.compare_target_dir, exist_ok=True)


def generate_condition_list():
    return [g.seed, g.val, optimization_methods_names[g.select_opt], mlmodel_names[g.select_ml], g.n_item, g.n_data, g.n_feature, g.n_user_available_x, get_noise()[g.noise_sigma], g.n_nearest]


def set_data(solver, X_train, Y_train, X_test, Y_test):
    for i in range(g.n_item):
        x_train, x_test, y_train, y_test = load_train_test_data(i)
        if x_train is None:
            if same_val_data_size:
                x_train, y_train = solver.problem.generate_dataset(i)
                x_test, y_test = solver.problem.generate_val_dataset(i)
            else:
                xs, c = solver.problem.generate_dataset(i)
                x_train, x_test, y_train, y_test = train_test_split(xs, c, train_size=train_rate, random_state=g.seed)
            save_train_test_data([x_train, x_test, y_train, y_test], i)

        X_train.append(x_train)
        Y_train.append(y_train)
        X_test.append(x_test)
        Y_test.append(y_test)


def set_val_data(solver, S_optval):
    for i in val_list:
        for j in range(g.n_item):
            s_ij = load_data("val", i*1000 + j)
            if s_ij is None:
                if j == 0:
                    s_ = solver.problem.generate_s(j, 1)
                else:
                    s_ = np.concatenate([s_, solver.problem.generate_s(j, 1)])
                save_data(s_, "val", i*1000 + j)
            else:
                s_ = s_ij
        S_optval.append(s_)


def set_init_x(solver, X_init, XS_train, C_train):
    for j in val_list:
        if not random_initialize_x:
            init_x = []
            for i in range(g.n_item):
                # print(np.argmax(C_train[i]))
                if g.select_problem == INEQUALITYCONSTRAINT:
                    max_c, max_id = -np.inf, -1
                    for j in range(g.n_data):
                        if solver.problem.check_penalty(XS_train[i][j]):
                            now_c = C_train[i][j]
                            if max_c < now_c:
                                max_c = now_c
                                max_id = j
                    if max_id == -1:
                        print("stop")
                        init_x.append(np.array([0.] * g.n_user_available_x))
                    else:
                        init_x.append(XS_train[i][max_id][:g.n_user_available_x])
                else:
                    init_x.append(XS_train[i][np.argmax(C_train[i])][:g.n_user_available_x])
            init_x = np.array(init_x)
        else:
            init_x = load_data("init_x", g.n_item * 1000 + j)
            if init_x is None:
                while True:
                    init_x = np.array([random_atob(solver.problem.min_bounds[i][j], solver.problem.max_bounds[i][j]) for j in g.user_available_x for i in range(g.n_item)]).reshape(-1, g.n_user_available_x)
                    # if solver.problem.check_penalty(init_x):
                    break
                save_data(init_x, "init_x", g.n_item * 1000 + j)
        X_init.append(init_x)


def save_train_test_data(data, i):
    for j, name in enumerate(["x_train", "x_test", "y_train", "y_test"]):
        save_data(data[j], name, i)


def save_data(data, type, i):
    file_name = generate_data_file_name(type, i)
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_train_test_data(i):
    res = []
    for name in ["x_train", "x_test", "y_train", "y_test"]:
        res.append(load_data(name, i))
    return res

def load_data(type, i):
    data_file = generate_data_file_name(type, i)
    if os.path.isfile(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        return data
    return None

def eval_all_f(fs, XS_train, C_train, XS_test, C_test):
    train_rmse, test_rmse = [], []
    for i, f in enumerate(fs):
        tr, ts = f.eval_performance(XS_train[i], C_train[i], XS_test[i], C_test[i], show=False)
        train_rmse.append(tr)
        test_rmse.append(ts)
    return train_rmse, test_rmse


def save_or_read_ml_info(train_rmse, test_rmse, learning_time, i):
    ori_i = g.n_item
    g.n_item = i
    set_unique_parameter()
    generate_result_save_directory()
    file_name = generate_mlinfo_name()
    if not os.path.exists(file_name):
        train_rmse = np.average(np.array(train_rmse))
        test_rmse = np.average(np.array(test_rmse))
        info = np.array([train_rmse, test_rmse, learning_time]).reshape(1, -1)
        pd.DataFrame(info, columns=['train_rmse', 'test_rmse', 'learning_rate']).to_csv(file_name, index=False)
        time = -1
    else:
        time = pd.read_csv(file_name)['learning_rate'].values[0]
    g.n_item = ori_i
    set_unique_parameter()
    generate_result_save_directory()
    return time
    

def check_optimization_prosess(f, problem, iter, now_x, now_y, history_x, history_y, direct_x=None, direct_y=None, n_density=100):
    if environment == DOCKER or environment == TSUBAME:
        show = False
    elif environment == LOCAL:
        show = True
    file_name_3d, file_name_cm, file_name_data = generate_process_image_name(iter=iter, is_3d=True)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x0", size = 16)
    ax.set_ylabel("x1", size = 16)
    ax.set_zlabel("obj", size = 16)

    x = np.linspace(problem.real_min_bounds[0][0], problem.real_max_bounds[0][0], n_density)
    y = np.linspace(problem.real_min_bounds[0][1], problem.real_max_bounds[0][1], n_density)

    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(n_density):
        for j in range(n_density):
            try:
                Z.append(f.predict([[y[i], x[j]]]))
            except:
                Z.append(f(np.array([y[i], x[j]])))
    Z = np.array(Z).reshape(n_density, -1)
    ax.plot_surface(Y, X, Z, cmap = "summer", alpha=alpha)
    ax.contour(Y, X, Z, colors = "black", offset = -1)
    if history_x.size != 0:
        ax.scatter(history_x[:, 0, 0], history_x[:, 0, 1], history_y, s = 40, c = "red")
    if direct_x is not None:
        ax.scatter(direct_x[0, 0], direct_x[0, 1], direct_y, s = 40, c = "m")
    ax.scatter(now_x[0, 0], now_x[0, 1], now_y, s = 40, c = "blue")
    if show:
        plt.show()
    else:
        fig.savefig(file_name_3d)
    plt.close()

    fig = plt.figure(figsize=(8, 6))

    partision_size = 10
    scale = n_density - 1
    start_x = problem.real_min_bounds[0][0]
    start_y = problem.real_min_bounds[0][1]
    point_ori = [scale * (i / partision_size) for i in range(partision_size + 1)]
    range_x = (problem.real_max_bounds[0][0] - problem.real_min_bounds[0][0])
    point_x = [round(range_x * (i / partision_size) + start_x, 1) for i in range(partision_size + 1)]
    range_y = (problem.real_max_bounds[0][1] - problem.real_min_bounds[0][1])
    point_y = [round(range_y * (i / partision_size) + start_y, 1) for i in range(partision_size + 1)]
    plt.xticks(point_ori, point_x)
    plt.yticks(point_ori, point_y)
    plt.xlabel("x1")
    plt.ylabel("x0")
    plt.imshow(Z, cmap="inferno", origin='upper')
    plt.title("Plot 2D array")
    plt.colorbar()

    rate = plot_colormap_rate_dic[g.select_data_type]
    if history_x.size != 0:
        x1, x0 = (history_x[:, 0, 1] - start_y) * rate, (history_x[:, 0, 0] - start_x) * rate
        # plt.scatter(history_x[:, 0, 1] * range_y - start_y, history_x[:, 0, 0] * range_x - start_x, s = 40, c = "red")
        plt.scatter(x1, x0, s = 40, c = "red")
    if direct_x is not None:
        x1, x0 = (direct_x[0, 1] - start_y) * rate, (direct_x[0, 0] - start_x) * rate
        # plt.scatter(direct_x[0, 1] * range_y - start_y, direct_x[0, 0] * range_x - start_x, s = 40, c = "m")
        plt.scatter(x1, x0, s = 40, c = "m")
    x1, x0 = (now_x[0, 1] - start_y) * rate, (now_x[0, 0] - start_x) * rate
    # plt.scatter(now_x[0, 1] * range_y - start_y, now_x[0, 0] * range_x - start_x, s = 40, c = "blue")
    plt.scatter(x1, x0, s = 40, c = "blue")
    if show:
        plt.show()
    else:
        fig.savefig(file_name_cm)
    plt.close()

def check_regression_shape(f, problem, s, x, y, show=False, is_3d=False, n_density=300):
    if not is_3d:
        feature0_x = (np.linspace(problem.real_min_bounds[0][0], problem.real_max_bounds[0][0], n_density)).reshape(n_density, -1)
        testest_x = feature0_x.copy()
        if g.n_user_available_x != 1:
            zero_x = np.zeros((n_density, g.n_user_available_x-1))
            testest_x = np.concatenate([feature0_x, zero_x], axis=1)
        if g.n_environment_s != 0:
            test_s = np.array([list(s)[0] for _ in range(n_density)])
            testest_x = np.concatenate([testest_x, test_s], axis=1)

        fig = plt.figure()

        true_y = [problem.data_generator.true_func(data) for data in feature0_x]
        plt.rcParams["font.size"] = 15
        plt.plot(feature0_x, true_y, linestyle="dashed", label="f(x)", c="limegreen")
        plt.scatter(x, y, label="データ", s=10)

        if g.select_opt == FRANKWOLFE:
            target_x = -4
            # plt.scatter(target_x, problem.data_generator.true_func(np.array([target_x])), color="black", s=60)
            # f.fit_xk(np.array([target_x]))
            # f.set_parameter()
            # tt_x = np.linspace(-5, 5, 100).reshape(100, -1)
            # dif_y = problem.data_generator.true_func(np.array([target_x])) - f.predict([target_x])
            # plt.plot(tt_x, f.predict(tt_x) + dif_y, label="提案手法", c="red", lw=2)
            
            # f.fit_xk(np.array([-3]))
            # f.set_parameter()
            # tt_x = np.linspace(-4, -2, 100).reshape(100, -1)
            # dif_y = problem.data_generator.true_func(np.array([-3])) - f.predict([-3])
            # plt.plot(tt_x, f.predict(tt_x) + dif_y, label="proposed method", c="red", lw=2)

            # f.fit_xk(np.array([-1.5]))
            # f.set_parameter()
            # tt_x = np.linspace(-2.5, -0.5, 100).reshape(100, -1)
            # dif_y = problem.data_generator.true_func(np.array([-1.5])) - f.predict([-1.5])
            # plt.plot(tt_x, f.predict(tt_x) + dif_y, c="red", lw=1.5)
            
            # f.fit_xk(np.array([0]))
            # f.set_parameter()
            # tt_x = np.linspace(-1, 1, 100).reshape(100, -1)
            # dif_y = problem.data_generator.true_func(np.array([0])) - f.predict([0])
            # plt.plot(tt_x, f.predict(tt_x) + dif_y, c="red", lw=1.5)

            plt.ylim(-10, 20)
        else:
            plt.plot(feature0_x, f.predict(testest_x), label="LightGBM", c="red", lw=2)
            # plt.ylim(-200,140)
        plt.legend(prop={"family":"MS Gothic"})
        plt.xlabel("x")
        plt.ylabel("y")
        if show:
            plt.show()
        else:
            file_name = generate_shape_image_name()
            fig.savefig(file_name)
        plt.close()
    else:
        file_name_3d, file_name_cm, file_name_data = generate_shape_image_name(is_3d=True)
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x[:, 0], x[:, 1], y)
        ax.set_xlabel("x0", size = 16)
        ax.set_ylabel("x1", size = 16)
        ax.set_zlabel("obj", size = 16)
        if show:
            plt.show()
        else:
            fig.savefig(file_name_data)
        plt.close()


        # Figureと3DAxeS
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x[:, 0], x[:, 1], y)

        # 軸ラベルを設定
        ax.set_xlabel("x0", size = 16)
        ax.set_ylabel("x1", size = 16)
        ax.set_zlabel("obj", size = 16)
        # (x,y)データを作成
        x = np.linspace(problem.real_min_bounds[0][0], problem.real_max_bounds[0][0], n_density)
        y = np.linspace(problem.real_min_bounds[0][1], problem.real_max_bounds[0][1], n_density)

        X, Y = np.meshgrid(x, y)
        Z = []
        for i in range(n_density):
            for j in range(n_density):
                try:
                    Z.append(f.predict(np.array([[x[i], y[j]]])))
                except:
                    Z.append(f([x[i], y[j]]))
        Z = np.array(Z).reshape(n_density, -1)
        ax.plot_surface(Y, X, Z, cmap = "summer")
        ax.contour(Y, X, Z, colors = "black", offset = -1)

        if show:
            plt.show()
        else:
            fig.savefig(file_name_3d)
        plt.close()

        fig = plt.figure(figsize=(8, 6))

        partision_size = 10
        scale = n_density - 1
        start_x = problem.real_min_bounds[0][0]
        start_y = problem.real_min_bounds[0][1]
        point_ori = [scale * (i / partision_size) for i in range(partision_size + 1)]
        range_x = (problem.real_max_bounds[0][0] - problem.real_min_bounds[0][0])
        point_x = [round(range_x * (i / partision_size) + start_x, 1) for i in range(partision_size + 1)]
        range_y = (problem.real_max_bounds[0][1] - problem.real_min_bounds[0][1])
        point_y = [round(range_y * (i / partision_size) + start_y, 1) for i in range(partision_size + 1)]

        plt.xticks(point_ori, point_x)
        plt.yticks(point_ori, point_y)
        plt.xlabel("x1")
        plt.ylabel("x0")
        plt.imshow(Z, cmap="inferno", origin='upper')
        plt.title("Plot 2D array")
        plt.colorbar()

        if show:
            plt.show()
        else:
            fig.savefig(file_name_cm)
        plt.close()



def create_gif():
    output_file = generate_gif_name()
    path_list = []
    
    for i in range(n_max_iteration):
        file_name_3d, file_name_cm, file_name_data = generate_process_image_name(iter=i, is_3d=True)
        path_list.append(file_name_cm)
 
    imgs = []
    for i in range(len(path_list)):
        img = Image.open(path_list[i])
        imgs.append(img)
    imgs[0].save(output_file, save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)


def adjust_cordinate_for_process(direct_x):
    if direct_x[0, 1] >= max_xs - delta:
        direct_x[0, 1] = max_xs - delta
    if direct_x[0, 0] >= max_xs - delta:
        direct_x[0, 0] = max_xs - delta
    return direct_x

def tic():
    g.start_time = time.time()

def toc():
    return time.time() - g.start_time


def tic2():
    g.start_time2 = time.time()

def toc2():
    return time.time() - g.start_time2

def random_atob(a, b):
    return (b - a) * np.random.rand() + a

def get_noise():
    if g.select_data_type == COMPLEX7:
        return Complex7.noises
    else:
        return BaseDataGeneration.noises


def print_start():
    ex = g.result_opt_dir + "/" + g.result_file_name
    print(ex)