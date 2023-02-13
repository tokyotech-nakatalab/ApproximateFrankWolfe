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
        # 2次の多項式特徴量のクラスをインスタンス化
        self.quadratic = PolynomialFeatures(
                               degree=2,                  # 多項式の次数
                               interaction_only=False,    # Trueの場合、ある特徴量を2乗以上した項が除かれる
                               include_bias=False,         # Trueの場合、バイアス項を含める
                               order='C'                  # 出力する配列の計算順序
                              )

    def set_parameter(self):
        self.linear_coef = self.mdl.coef_
        self.linear_ic = self.mdl.intercept_

    def set_parameter_for_opt(self, s):
        self.coefs = self.linear_coef
        self.ic = self.linear_ic

    def fit(self, x, y):
        # データ演算・変換
        X_quad_train  = self.quadratic.fit_transform(x)
        # 回帰モデルに上記データを適合
        self.mdl.fit(X_quad_train, y)

    def predict(self, xs):
        xs = self.quadratic.fit_transform(xs)
        return xs @ self.linear_coef + self.linear_ic