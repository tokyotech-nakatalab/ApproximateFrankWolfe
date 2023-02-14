from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *
from machine_learning.tools.ann import *
from utility.tool import tic2, toc2

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
        tic2()
        nearest_idx, distances = self.ann.predict(x_k, g.n_nearest)
        g.search_time += toc2()
        nearest_x_list = self.x[nearest_idx]
        # u, s, vh = svd(nearest_x_list)
        average_x = np.mean(nearest_x_list, 0)
        tic2()
        # u, s, vh = svd(nearest_x_list-average_x)
        s = -1
        g.svd_time += toc2()
        # print(f"最小特異値:{np.min(s)}")
        min_s = np.min(s)
        min_s = 1 / min_s
        tic2()
        diam = np.max(distances)
        g.culcd_time += toc2()
        # print(f"平均距離:{diam}")

        tic2()
        self.mdl.fit(nearest_x_list, self.y[nearest_idx])
        g.fit_time += toc2()
        return distances, min_s, diam


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