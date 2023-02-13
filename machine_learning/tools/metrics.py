from utility.module import *


def RMSE(y, y_hat):
    #RMSE計算
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    return rmse