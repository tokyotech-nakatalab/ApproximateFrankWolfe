from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *

class MyLinearRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = LINEARREGRESSION
        self.mdl = LinearRegression()

    def set_parameter(self):
        self.linear_coef = self.mdl.coef_
        self.linear_ic = self.mdl.intercept_

    def set_parameter_for_opt(self, s):
        ic = self.linear_coef[g.n_user_available_x:] @ s
        self.coefs = self.linear_coef[:g.n_user_available_x]
        self.ic = ic + self.linear_ic

    def predict(self, xs):
        return xs @ self.linear_coef + self.linear_ic