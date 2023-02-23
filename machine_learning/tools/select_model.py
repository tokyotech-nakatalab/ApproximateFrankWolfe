from machine_learning.model.ann_linear_regression import MyAnnLinearRegression
from machine_learning.model.knn_linear_regression import MyKnnLinearRegression
from utility.constant import *
from utility.setting import *
from utility.tool import *

from machine_learning.model.linear_regression import *
from machine_learning.model.supportvector_regression import *
from machine_learning.model.random_forest import *
from machine_learning.model.cga2m_regression import *
from machine_learning.model.lightgbm_regression import *
from machine_learning.model.neural_network import *
from machine_learning.model.weighted_linear_regression import *
from machine_learning.model.polynomial_regression import *
from machine_learning.model.gauss_process_regression import *
import utility.gloabl_values as g


def select_model(x_train, y_train, i):
    model = load_model(i)
    # model = None
    if not model is None:
        model.is_saved_flg = True
        return model
    else:
        if g.select_ml == LINEARREGRESSION:
            return MyLinearRegression(i)
        elif g.select_ml == SVRLINEAR:
            return MySupportVectorRegression(i, SVRLINEAR)
        elif g.select_ml == SVRPOLY:
            return MySupportVectorRegression(i, SVRPOLY)
        elif g.select_ml == SVRGAUSS:
            return MySupportVectorRegression(i, SVRGAUSS)
        elif g.select_ml == RANDOMFOREST:
            return MyRandomForest()
        elif g.select_ml == LIGHTGBM:
            return MyLightGBM(i)
        elif g.select_ml == NEURALNETWORK:
            return MyNeuralNetwork(i)
        elif g.select_ml == CGA2M:
            x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, train_size=eval_rate, random_state=g.seed)
            return MyCGA2M(i, x_train, y_train, x_eval, y_eval)
        elif g.select_ml == WEIGHTEDLINEARREGRESSION:
            return MyWeightedLinearRegression(i)
        elif g.select_ml == ANNLINEARREGRESSION:
            return MyAnnLinearRegression(i)
        elif g.select_ml == KNNLINEARREGRESSION:
            return MyKnnLinearRegression(i)
        elif g.select_ml == POLYNOMIALREGRESSION:
            return MyPolynomialRegression(i)
        elif g.select_ml == GAUSSIANPROCESSREGRESSION:
            return MyGaussianProcessRegression(i)


def model_fit(mymdl, x, y, i):
    if not mymdl.is_saved_flg:
        # 学習
        if g.select_ml == LINEARREGRESSION:
            mymdl.fit(x, y)
        elif g.select_ml == POLYNOMIALREGRESSION:
            if search_hyper_paramerter:
                mymdl.grid_search(x, y)
            mymdl.fit(x, y) 
        elif g.select_ml == SVRLINEAR or g.select_ml == SVRPOLY or g.select_ml == SVRGAUSS:
            # mymdl.bayse_search(x, y)
            if search_hyper_paramerter:
                mymdl.grid_search(x, y)
            mymdl.fit(x, y)  
        elif g.select_ml == RANDOMFOREST or g.select_ml == LIGHTGBM:
            if search_hyper_paramerter:
                mymdl.bayse_search(x, y)
            mymdl.fit(x, y)    
        elif g.select_ml == NEURALNETWORK:
            if search_hyper_paramerter:
                mymdl.grid_search(x, y)
            mymdl.fit(x, y)
        elif g.select_ml == CGA2M:
            mymdl.fit()
        elif g.select_ml == WEIGHTEDLINEARREGRESSION:
            mymdl.set_data(x, y)
        elif g.select_ml == ANNLINEARREGRESSION:
            mymdl.set_data(x, y)
        elif g.select_ml == KNNLINEARREGRESSION:
            mymdl.set_data(x, y)
        
        # 保存
        if not g.select_ml in [ANNLINEARREGRESSION, KNNLINEARREGRESSION, WEIGHTEDLINEARREGRESSION]:
            save_model(mymdl, i)
    return mymdl


def create_fit_llr(i, x, y, max_bound, min_bound):
    mdl = MyLocalLinearRegression(i, max_bound, min_bound)
    mdl.bayse_search(x, y)
    mdl.fit(x, y)
    return mdl


def save_model(mymdl, i):
    file_name = generate_model_file_name(i)
    with open(file_name, "wb") as f:
        pickle.dump(mymdl, f)

def load_model(i):
    model_file = generate_model_file_name(i)

    if os.path.isfile(model_file):
        #読み出し
        with open(model_file, "rb") as f:
            mymdl = pickle.load(f)
        return mymdl
    return None