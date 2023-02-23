from utility.module import *
from utility.constant import *

from machine_learning.model.base_model import *
from machine_learning.tools.metrics import *
from machine_learning.tools.tree_visualize import *


class MyPolynomialRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = POLYNOMIALREGRESSION
        self.mdl = LinearRegression()
        self.gsr_flg = False

    def set_parameter(self):
        self.linear_coef = self.mdl.coef_
        self.linear_ic = self.mdl.intercept_

    def set_parameter_for_opt(self, s):
        self.coefs = self.linear_coef
        self.ic = self.linear_ic

    def fit(self, x, y):
        # データ演算・変換
        # 回帰モデルに上記データを適合
        if self.gsr_flg:
            self.quadratic = PolynomialFeatures(
                               degree=self.best_poly,                  # 多項式の次数
                               interaction_only=False,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
                               include_bias=False,         # Trueの場合、バイアス項を含める
                               order='C'                  # 出力する配列の計算順序
                              )
            X_quad_train  = self.quadratic.fit_transform(x)
        else:
            # 2次の多項式特徴量のクラスをインスタンス化
            self.quadratic = PolynomialFeatures(
                               degree=2,                  # 多項式の次数
                               interaction_only=False,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
                               include_bias=False,         # Trueの場合、バイアス項を含める
                               order='C'                  # 出力する配列の計算順序
                              )
            X_quad_train  = self.quadratic.fit_transform(x)
        self.mdl.fit(X_quad_train, y)

    def predict(self, xs):
        xs = self.quadratic.fit_transform(xs)
        return xs @ self.linear_coef + self.linear_ic

    def grid_search(self, x, y):
        candidate = [2, 3]
        min_rmse = np.inf
        self.best_poly = 0
        kf = KFold(n_splits=n_split)
        for poly in candidate:
            serach_quadratic = PolynomialFeatures(
                                degree=poly,                  # 多項式の次数
                                interaction_only=False,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
                                include_bias=False,         # Trueの場合、バイアス項を含める
                                order='C'                  # 出力する配列の計算順序
                                )
            rmse_score = 0
            for index in kf.split(x):
                train_index, test_index = index[0], index[1]
                X_quad_train = serach_quadratic.fit_transform(x[train_index])
                mdl = LinearRegression()
                mdl.fit(X_quad_train, y[train_index])
                y_pred = mdl.predict(serach_quadratic.fit_transform(x[test_index]))
                rmse_score += RMSE(y[test_index], y_pred)
            if min_rmse > rmse_score:
                min_rmse = rmse_score
                self.best_poly = poly
        self.gsr_flg = True

