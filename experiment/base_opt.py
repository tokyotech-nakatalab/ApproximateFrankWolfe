from utility.module import *
from utility.setting import *
from experiment.generate_data import *
import utility.gloabl_values as g

class BaseOfData():
    def __init__(self) -> None:
        if g.select_data_type == SINX0:
            self.data_generator = SinX0()
        elif g.select_data_type == SINX0MOUNT2:
            self.data_generator = SinX0Mount2()
        elif g.select_data_type == SINX01MOUNT2:
            self.data_generator = SinX01Mount2()
        elif g.select_data_type == ROSENBROCK:
            self.data_generator = RosenBrock()
        elif g.select_data_type == ACKELY:
            self.data_generator = Ackley()
        elif g.select_data_type == XSQUARE:
            self.data_generator = XSquare()
        elif g.select_data_type == COMPLEX7:
            self.data_generator = Complex7()
        elif g.select_data_type == MOUNT2:
            self.data_generator = Mount2()
        elif g.select_data_type == RASTRIGIN:
            self.data_generator = Rastrigin()
        elif g.select_data_type == LOGX:
            self.data_generator = LogX()
        elif g.select_data_type == LOGX2:
            self.data_generator = LogX2()
        elif g.select_data_type == REAL:
            self.data_generator = AdvertisingData()
        self.calc_c = self.data_generator.true_func
        self.noise = self.data_generator.noise

    def generate_dataset(self, i):
        x = self.generate_x(i, g.n_data)
        s = self.generate_s(i, g.n_data)
        xs = np.concatenate([x, s], axis=1)
        c = []
        for j in range(g.n_data):
            true_c = self.calc_c(xs[j], i)
            observed_c = true_c + self.noise(xs[j])
            c.append(observed_c)
        c = np.array(c)
        return xs, c

    def generate_val_dataset(self, i):
        x = self.generate_x(i, n_val_data)
        s = self.generate_s(i, n_val_data)
        xs = np.concatenate([x, s], axis=1)
        c = []
        for j in range(n_val_data):
            true_c = self.calc_c(xs[j], i)
            observed_c = true_c + self.noise(xs[j])
            c.append(observed_c)
        c = np.array(c)
        return xs, c


    def generate_x(self, i, n):
        if is_x_normal: # 正規分布でサンプリング
            sample = np.zeros((n, g.n_user_available_x))
            for j in g.user_available_x:
                for k in range(n):
                    while True:
                        sam = np.random.normal(loc = self.data_generator.x_ast[j], scale = x_scale, size = 1)
                        if not (sam < self.min_bounds[i][j] or sam > self.max_bounds[i][j]):
                            sample[k, j] = sam
                            break

        else: # 一様分布でサンプリング
            sample = np.random.random_sample((n, g.n_user_available_x))
            for j in g.user_available_x:
                sample[:, j] = (self.max_bounds[i][j] - self.min_bounds[i][j]) * sample[:, j] + self.min_bounds[i][j]
        return sample

    def generate_s(self, i, n):
        sample = np.random.random_sample((n, g.n_environment_s))
        for j in g.environment_s:
            j_index = j - g.n_user_available_x
            sample[:, j_index] = (self.max_bounds[i][j] - self.min_bounds[i][j]) * sample[:, j_index] + self.min_bounds[i][j]
        return sample

