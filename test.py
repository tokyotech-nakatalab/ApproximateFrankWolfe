import numpy as np

def calc(x):
    # ans = 0
    # for j in range(x.size//2):
    #     ans += (1 + (x[2*j] + x[2*j+1] + 1)**2 * (19 - 14*x[2*j] + 3*(x[2*j]**2) - 14*x[2*j+1] + 6*x[2*j]*x[2*j+1] + 3*(x[2*j+1]**2))) * (30 + (2*x[2*j] - 3*x[2*j+1])**2 * (18 - 32*x[2*j] + 12*(x[2*j]**2) + 48*x[2*j+1] - 36*x[2*j]*x[2*j+1] + 27*(x[2*j+1]**2)))
    # return ans
    ans = 0
    for j in range(x.size//2):
        # ans += (x[2*j] + 2*x[2*j+1] - 7)**2 + (2*x[2*j] + x[2*j+1] - 5)**2
        ans += (1.5 - x[2*j] + x[2*j] * x[2*j+1])**2 + (2.25 - x[2*j] + x[2*j] * (x[2*j+1] ** 2))**2 + (2.625 - x[2*j] + x[2*j] * (x[2*j+1]**3))**2
    return -ans

print(calc(np.array([3, 0.5])))