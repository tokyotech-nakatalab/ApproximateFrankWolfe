from utility.module import *
from utility.setting import base_th_distance
import utility.gloabl_values as g
from machine_learning.tools.knn import BaseKNN

class FaissANN(BaseKNN):
    def fit(self, x):
        x = x.astype('float32')
        nlist = 100
        quantizer = faiss.IndexFlatL2(x.shape[1])
        self.index = faiss.IndexIVFFlat(quantizer, x.shape[1], nlist)
        self.index.train(x)
        self.index.add(x)

    def predict(self, xq, k):
        xq = xq.reshape(1, -1).astype('float32')
        distances, ids = self.index.search(xq, k=k)
        ids, distances = np.array(ids), np.array(distances)
        if g.n_nearest == g.n_nearest:
            ids2, distances2 = ids[distances < self.distance_threshold], distances[distances < self.distance_threshold]
            if ids2.size >= 30:
                ids, distances = ids2, distances2
        return ids.ravel(), distances.ravel()


class AnnoyANN(BaseKNN):
    def fit(self, x):
        # 最初にvectorサイズを入れる
        self.t = AnnoyIndex(x.shape[1], metric='euclidean')
        for i, v in enumerate(x):
            # indexつけて一個ずつ入れる必要がある。これが遅そう。。
            self.t.add_item(i, v)
        self.t.build(10) # 10 trees

    def predict(self, xq, k):
        ids, distances = self.t.get_nns_by_vector(xq, k, include_distances=True) # id:ベクトルid，distance:距離
        ids, distances = np.array(ids), np.array(distances)
        # if g.n_nearest == n_nearest:
        #     ids, distances = ids[distances < self.distance_threshold], distances[distances < self.distance_threshold]
        return ids, distances


class NmslibANN(BaseKNN):
    def fit(self, x):
        self.x = x
        # Annoy同様にデータを入れてbuildする。Numpy配列で入れられる。
        self.index = nmslib.init(method='hnsw', space='l2')
        self.index.addDataPointBatch(x)
        self.index.createIndex({'post': 2}, print_progress=True)

    def predict(self, xq, k):
        # 基本的にAnnoy同様に一件ずつ検索して、返却される。
        ids, distances = self.index.knnQuery(xq, k=k)
        return ids, distances