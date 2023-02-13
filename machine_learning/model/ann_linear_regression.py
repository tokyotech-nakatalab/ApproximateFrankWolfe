from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *
from machine_learning.tools.ann import *

class MyAnnLinearRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = LINEARREGRESSION
        self.mdl = LinearRegression()

    def set_data(self, x, y):
        self.x = x
        self.y = y
        if ann_library == ANNOY:
            self.ann = AnnoyANN()
        elif ann_library == NMSLIB:
            self.ann = NmslibANN()
        elif ann_library == FAISS:
            self.ann = FaissANN()
        self.ann.fit(x)

    def fit_xk(self, x_k):
        nearest_idx, distances = self.ann.predict(x_k, g.n_nearest)
        self.mdl.fit(self.x[nearest_idx], self.y[nearest_idx])
        return distances

    def set_parameter(self):
        try:
            self.linear_coef = self.mdl.coef_
            self.linear_ic = self.mdl.intercept_
        except:
            return

    def set_parameter_for_opt(self, s):
        ic = self.linear_coef[g.n_user_available_x:] @ s
        self.coefs = self.linear_coef[:g.n_user_available_x]
        self.ic = ic + self.linear_ic

    def predict(self, xs):
        return xs @ self.linear_coef + self.linear_ic

    def eval_performance(self, x_train, y_train, x_test, y_test, show=True):
        if show:
            print(f'Train RMSE :-')
            print(f'Test RMSE :-')
        return 0, 0