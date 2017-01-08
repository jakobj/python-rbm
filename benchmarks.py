import numpy as np


def lxor(n):
    v = np.random.randint(0, 2, size=(n, 3))
    v[:, 2] = np.logical_xor(v[:, 0], v[:, 1])
    return v


def land(n):
    v = np.random.randint(0, 2, size=(n, 3))
    v[:, 2] = np.logical_and(v[:, 0], v[:, 1])
    return v
