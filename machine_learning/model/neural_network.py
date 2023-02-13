from utility.module import *
from utility.constant import *
from utility.setting import *

from machine_learning.model.base_model import *


class MyNeuralNetwork(BaseMyModel):
    def __init__(self, i) -> None:
        super().__init__(i)
        self.name = NEURALNETWORK
        self.bayse_flg = False
        self.base_params = {"random_state": g.seed}


    def fit(self, x, y):
        self.mdl = MLPRegressor(**self.base_params)
        self.mdl.fit(x, y)
