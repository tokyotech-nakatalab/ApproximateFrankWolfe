from utility.module import *
from utility.setting import *

from machine_learning.model.base_model import *

class MyGaussianProcessRegression(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = GAUSSIANPROCESSREGRESSION
        kk = gp.kernels.RBF() + gp.kernels.WhiteKernel()
        self.mdl = gp.GaussianProcessRegressor(kernel=kk)

    def fit(self, x, y):
        self.mdl.fit(x, y)
        print("Kernel: ", self.mdl.kernel_)

    def predict(self, x):
        return self.mdl.predict(x, return_std=True)

    def eval_performance(self, x_train, y_train, x_test, y_test, show=True):
        if show:
            print(f'Train RMSE :-')
            print(f'Test RMSE :-')
        return 0, 0
