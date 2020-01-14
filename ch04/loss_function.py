import numpy as np


def sum_of_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 2が正解
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 2が最大値
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]  # 7が最大値

sum_of_squared_error(np.array(y1), np.array(t))
sum_of_squared_error(np.array(y2), np.array(t))


def cross_entropy_error2(y, t):
    if y.ndim == 1:  # データが1次元の場合
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 高次元の場合
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
