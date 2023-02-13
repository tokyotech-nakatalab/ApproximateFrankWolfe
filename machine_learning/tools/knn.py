from utility.module import *
from utility.setting import base_th_distance
import utility.gloabl_values as g

class BaseKNN():
    def __init__(self) -> None:
        self.distance_threshold = (base_th_distance ** 2) * g.n_feature

    def fit(self, x):
        pass

    def predict(self, xq):
        pass


class SckitLearnKNN(BaseKNN):
    def fit(self, x):
        self.knn_model = NearestNeighbors(n_neighbors=g.n_nearest, algorithm='ball_tree').fit(x) 


    def predict(self, xq, k):
        xq = xq.reshape(-1, g.n_feature)
        distances, indices = self.knn_model.kneighbors(xq) 
        distances, ids = distances[:k].ravel(), indices[0][:k]         
        # if g.n_nearest == n_nearest:
        #     ids, distances = ids[distances < self.distance_threshold], distances[distances < self.distance_threshold]
        return ids, distances