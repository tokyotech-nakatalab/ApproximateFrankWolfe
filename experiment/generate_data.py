from utility.module import *
from utility.setting import min_xs, max_xs
import utility.gloabl_values as g


class BaseDataGeneration():
    noises = [0, 1, 3, 5, 7, 10]

    def __init__(self) -> None:
        self.feature_bounds()
        self.set_x_ast()

    def true_func(self, i=None):
        pass

    def feature_bounds(self):
        pass

    def adjust_x(self, x, i):
        x[x < 0] = x[x < 0] * self.min_rates[i][x < 0]
        x[x > 0] = x[x > 0] * self.max_rates[i][x > 0]
        return x

    def noise(self, x):
        return np.random.normal(loc=0, scale=self.noises[g.noise_sigma], size=1).item()


class XSquare(BaseDataGeneration):
    def true_func(self, x, i=0):
        return -np.sum(x**2)
    
    def set_x_ast(self):
        self.x_ast = [0.] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]

class LogX(BaseDataGeneration):
    def true_func(self, x, i=0):
        # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if g.n_feature == 10:
            a = np.array([5, 4, 6, 5, 8, 6, 4, 6, 6, 7])
            b = np.array([4, 3, 3, 6, 6, 5, 4, 5, 3, 5])
            # c = np.array([6, 6, 5, 5, 7, 4, 5, 6, 4, 6])
            c = np.array([8, 5, 3, 7, 4, 6, 3, 4, 7, 8])
        elif g.n_feature == 30:
            a = np.array([5, 4, 6, 5, 8, 6, 4, 6, 6, 7, 
                          6, 5, 5, 4, 5, 7, 3, 5, 6, 4, 
                          5, 6, 5, 4, 5, 6, 8, 5, 6, 4])
        if np.sum(x) == 0:
            return 10 ** (-5)
        else:
            ans = g.coef * (np.log(a @ x) + np.log(b @ x) + np.log(c @ x))
            # return ans - 10**7
            return ans
            # return g.coef * (np.log(a @ x) +b @ x - 10 * g.coef
    
    def set_x_ast(self):
        self.x_ast = [0.] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]


class LogX2(BaseDataGeneration):
    def true_func(self, x, i=0):
        # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if g.n_feature == 10:
            a = np.array([5, 4, 6, 5, 8, 6, 4, 6, 6, 7])
        elif g.n_feature == 30:
            a = np.array([5, 4, 6, 5, 8, 6, 4, 6, 6, 7, 
                          6, 5, 5, 4, 5, 7, 3, 5, 6, 4, 
                          5, 6, 5, 4, 5, 6, 8, 5, 6, 4])
        coef = 10
        if np.sum(x) == 0:
            return 10 ** (-5)
        else:
            return coef * np.log(a @ (x**2)) - 10 * coef
    
    def set_x_ast(self):
        self.x_ast = [0.] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]


class SinX0(BaseDataGeneration):
    def true_func(self, x, i=None):
        ans = np.copy(x)
        ans[ans > 0] = (ans[ans > 0] - 1)**2
        ans[ans < 0] = (ans[ans < 0] + 1/2)**2 + 3/4
        ans = np.sum(ans)
        # ans = ans / g.n_feature
        return -ans

    def set_x_ast(self):
        self.x_ast = [1.] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]


class SinX0Mount2(BaseDataGeneration):
    def true_func(self, x, i=None):
        ans = np.copy(x)
        ans[ans > 0] = (ans[ans > 0] - 3/2)**2
        ans[ans < 0] = (ans[ans < 0] + 1/2)**2 + 2
        ans = np.sum(ans)
        # ans = ans / g.n_feature
        return -ans

    def set_x_ast(self):
        self.x_ast = [3/2] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]



class SinX01Mount2(BaseDataGeneration):
    def true_func(self, x, i=None):
        ans = self.mount2_sin_func(x[0])
        ans2 = self.mount2_sin_func(x[1])
        return ans + ans2

    def mount2_sin_func(sekf, x):
        ans = 10 * np.sin(x)
        if isinstance(ans, np.floating):
            if x > 0:
                ans = 15 * np.sin(x)
        else:
            ans[x > 0] = 15 * np.sin(x)
        return ans

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]


class Mount2(BaseDataGeneration):
    def true_func(self, x, i=None):
        ans = 1/8* (x**4) - 1/6 * (x**3) - 3/2 * (x**2) + 63/8
        # a = 2592 * 1/2
        # ans = (a * 1/4 * (x**4) + 1/45 * (x**3) - 1/8 * (x**2)) - 1863/40000 * a 
        # ans = a * (21/2 * (x**4) - 1/3 * (x**3) - 1/2 * (x**2)) + 19/2592 * a 
        ans = -ans
        # ans = -ans
        # ans = np.sum(ans) / g.n_feature
        return ans

    def set_x_ast(self):
        self.x_ast = [3] * g.n_feature
        # self.x_ast = [1/6] * g.n_feature 

    def feature_bounds(self):
        self.min_bounds = [[min_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]
        self.max_bounds = [[max_xs for _ in range(g.n_feature)] for __ in range(g.n_item)]


class RosenBrock(BaseDataGeneration):
    def true_func(self, x, i=0):
        x = self.adjust_x(x, 0)
        return -1 *((1 - x[0])**2 + 100 * ((x[1] - x[0]**2)**2))

    def feature_bounds(self):
        self.original_min_bounds = np.array([[-1.5] * g.n_feature for _ in range(g.n_item)])
        self.original_max_bounds = np.array([[1.5] * g.n_feature for _ in range(g.n_item)])
        self.min_bounds = np.array([[min_xs] * g.n_feature for _ in range(g.n_item)])
        self.max_bounds = np.array([[max_xs] * g.n_feature for _ in range(g.n_item)])
        self.min_rates = np.abs(self.original_min_bounds / min_xs)
        self.max_rates = np.abs(self.original_max_bounds / max_xs)


class Ackley(BaseDataGeneration):
    def true_func(self, x, i=0):
        base = 20
        x = self.adjust_x(x, i)
        t1 = base
        t2 = - base * np.exp(- 0.2 * np.sqrt(1.0 / g.n_feature * np.sum(x ** 2)))
        t3 = np.e
        t4 = - np.exp(1.0 / g.n_feature * np.sum(np.cos(2 * np.pi * x)))
        return -1 * (t1 + t2 + t3 + t4)

    def set_x_ast(self):
        self.x_ast = [0.] * g.n_feature 

    def feature_bounds(self):
        self.original_min_bounds = np.array([[-32.768] * g.n_feature for _ in range(g.n_item)])
        self.original_max_bounds = np.array([[32.768] * g.n_feature for _ in range(g.n_item)])
        self.min_bounds = np.array([[min_xs] * g.n_feature for _ in range(g.n_item)])
        self.max_bounds = np.array([[max_xs] * g.n_feature for _ in range(g.n_item)])
        self.min_rates = np.abs(self.original_min_bounds / min_xs)
        self.max_rates = np.abs(self.original_max_bounds / max_xs)


class Rastrigin(BaseDataGeneration):
    def true_func(self, x, i=0):
        x = self.adjust_x(x, i)
        ans = 10 * g.n_feature + np.sum(x**2 - 10 * np.cos(2*np.pi*x))
        return -1 * ans

    def set_x_ast(self):
        self.x_ast = [0.] * g.n_feature 

    def feature_bounds(self):
        self.original_min_bounds = np.array([[-5.12] * g.n_feature for _ in range(g.n_item)])
        self.original_max_bounds = np.array([[5.12] * g.n_feature for _ in range(g.n_item)])
        self.min_bounds = np.array([[min_xs] * g.n_feature for _ in range(g.n_item)])
        self.max_bounds = np.array([[max_xs] * g.n_feature for _ in range(g.n_item)])
        self.min_rates = np.abs(self.original_min_bounds / min_xs)
        self.max_rates = np.abs(self.original_max_bounds / max_xs)


class Complex7(BaseDataGeneration):
    noises = [0, 0.2, 0.5, 1.0, 2.0]

    
    def true_func(self, x, i=None):
        x = self.adjust_x(x, 0)
        ans = np.exp(- 1/3 * x[1]**2) +  np.sin(2* np.pi * x[2]) - 1/3 * x[2]*x[4] + 1/3 * np.exp(1/8*(1/2*x[3] + x[4] - 1/4*x[5])**2) 
        return ans


    def feature_bounds(self):
        self.original_min_bounds = np.array([[-2.] * g.n_feature for _ in range(g.n_item)])
        self.original_max_bounds = np.array([[2.] * g.n_feature for _ in range(g.n_item)])
        self.min_bounds = np.array([[min_xs] * g.n_feature for _ in range(g.n_item)])
        self.max_bounds = np.array([[max_xs] * g.n_feature for _ in range(g.n_item)])
        self.min_rates = np.abs(self.original_min_bounds / min_xs)
        self.max_rates = np.abs(self.original_max_bounds / max_xs)

    def noise(self, x):
        x = self.adjust_x(x, 0)
        ns = self.noises[g.noise_sigma]
        if ns == 0.:
            return 0.
        else:
            return (1 + ns * np.random.normal(size=1).item())*x[0]


class AdvertisingData(BaseDataGeneration):
    def set_x_ast(self):
        self.x_ast = [0] * g.n_feature

    def feature_bounds(self):
        self.original_min_bounds = np.array([[0, 0, 0]])
        self.original_max_bounds = np.array([[218.45, 36.55, 45.1]])
        self.min_bounds = self.original_min_bounds
        self.max_bounds = self.original_max_bounds
