import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel
from mpl_toolkits.mplot3d import Axes3D

from utility.module import *
from utility.setting import *
from experiment.base_opt import *
from solvers.base_optimization_method import *

class BayseOptimization(BaseOptimizationMethod):
    def optimize(self, fs, new_s, init_x):
        # ガウス過程回帰モデルによる予測値の計算
        temp = np.arange(self.problem.min_bounds[0], self.problem.max_bounds[0] + 0.01, 0.01)
        X = []
        
        G, SD = fs[0].model.predict(X, return_std=True)

        # 次の実験点の提案
        # 獲得関数の計算
        A, index = fs[0].aq(G, SD, len(X))
        user_x = X[index]

        x, obj, true_obj = self.evaluate_result(self.problem, fs, user_x, new_s)

        self.renew_best(x, obj, true_obj)
        return self.best_x, self.best_obj, self.best_true_obj

class BayseOptimizationMu(BayseOptimization):
    #獲得関数
    def aq(self, mu, sigma, N):
        a = mu + 3*sigma # a = mu + k * sigmaでkを実験回数で変更してもよい
        i = np.argmin(a)
        return a, i

class BayseOptimizationLCB(BayseOptimization):
    #獲得関数
    def aq(self, mu, sigma, N):
        a = mu
        i = np.argmin(a)
        return a, i
