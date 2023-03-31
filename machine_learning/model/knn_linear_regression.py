from machine_learning.tools.knn import SckitLearnKNN
from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *
from machine_learning.tools.ann import *
from utility.tool import tic2, toc2

class MyKnnLinearRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = LINEARREGRESSION
        self.mdl = LinearRegression()
        self.nearest_history = {}

    def set_data(self, x, y):
        self.x = x
        self.y = y
        self.knn = SckitLearnKNN()
        self.knn.fit(x)

    def fit_xk(self, x_k):
        tic2()
        nearest_idx, distances = self.knn.predict(x_k, g.n_nearest)
        try:
            ans = self.nearest_history[tuple(nearest_idx)]
            return ans, nearest_idx, True
        except:
            g.search_time += toc2()
            nearest_x_list = self.x[nearest_idx]
            average_x = np.mean(nearest_x_list, 0)
            tic2()
            # u, s, vh = svd(nearest_x_list)
            # u, s, vh = svd(nearest_x_list-average_x)
            s = -1
            g.svd_time += toc2()
            # print(f"最小特異値:{np.min(s)}")
            min_s = np.min(s)
            # min_s = 1 / min_s
            tic2()
            diam = np.max(distances)
            g.culcd_time += toc2()
            # print(f"平均距離:{diam}")

            tic2()
            self.mdl.fit(nearest_x_list, self.y[nearest_idx])
            g.fit_time += toc2()
            self.nearest_history[tuple(nearest_idx)] = (distances, min_s, diam)
            return (distances, min_s, diam), nearest_idx, False

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

def distance(x, y):
    dist = np.sqrt(np.sum(x - y)**2)
    return dist

def random_sample_distance(nearest_x_list):
    dist_list = []
    idx_list = np.array(list(range(nearest_x_list.shape[0])))
    for i in range(5):
        pair = np.random.choice(idx_list, 2, replace=False)
        dist_list.append(distance(nearest_x_list[pair[0]], nearest_x_list[pair[1]]))
    return sum(dist_list) / len(dist_list)

def distance(x, y):
    dist = np.sqrt(np.sum(x - y)**2)
    return dist


def calc_diameter(nearest_x_list):
    diam = -1
    for i in range(nearest_x_list.shape[0]):
        for j in range(nearest_x_list.shape[0]):
            if i != j :
                temp_diam = distance(nearest_x_list[i], nearest_x_list[j])
                if diam < temp_diam:
                    diam = temp_diam
    return diam