from machine_learning.tools.metrics import *
import utility.gloabl_values as g

class BaseMyModel():
    def __init__(self, i) -> None:
        self.f_index = i
        self.is_saved_flg = False

    def fit(self, x, y):
        self.mdl.fit(x, y)

    def set_parameter(self):
        pass

    def predict(self, x):
        return self.mdl.predict(x)

    def eval_performance(self, x_train, y_train, x_test, y_test, show=True):
        try:
            y_hat = self.mdl.predict(x_train)
        except:
            y_hat = self.predict(x_train)
        tr_rmse = RMSE(y_train, y_hat)
        if show:
            print('Train RMSE : {:.3f}'.format(tr_rmse))

        try:
            y_hat = self.mdl.predict(x_test)
        except:
            y_hat = self.predict(x_test)
        ts_rmse = RMSE(y_test, y_hat)
        if show:
            print('Test RMSE : {:.3f}'.format(ts_rmse))
        return tr_rmse, ts_rmse
     
     
    def local_linear_regression(self, x_k):
        exp = self.explainer.explain_instance(x_k, self.mdl.predict, num_features=g.n_feature)
        # print(exp)
        # exp.as_pyplot_figure() #discretized_feature_names: condition ,local_exp: coef
        return exp.local_exp[1], exp.intercept[0]



def add_intercept(x):
    try:
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    except:
        x = np.concatenate([np.ones(1), x], axis=0)
    return x

def kernel(x, x_k, sigma):
    x_2 = (x - x_k) @ (x - x_k)
    # k = np.exp(-1/(2 *(g.weight_sigma**2)) * x_2)
    k = np.exp(-1/(2 *(sigma**2)) * x_2)
    return k