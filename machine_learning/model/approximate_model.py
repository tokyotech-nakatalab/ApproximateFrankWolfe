from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import BaseMyModel


class MyApproximateModel(BaseMyModel):
    def __init__(self, f, xk, s) -> None:
        self.name = LINEARREGRESSION
        xs = np.concatenate([xk, s])
        coef, intercept = f.local_linear_regression(xs)
        self.linear_coef = [0.] * xs.size
        for i in range(len(coef)):
            feature_id = coef[i][0]
            feature_coef = coef[i][1]
            self.linear_coef[feature_id] = feature_coef
        self.linear_coef = np.array(self.linear_coef)
        self.linear_ic = intercept

    def set_parameter_for_opt(self, s):
        ic = self.linear_coef[g.n_user_available_x:] @ s
        self.coefs = np.round(self.linear_coef[:g.n_user_available_x], 3)
        self.ic = np.round(ic + self.linear_ic, 3)

    def predict(self, xs):
        return self.linear_coef @ xs + self.linear_ic