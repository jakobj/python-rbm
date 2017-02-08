# -*- coding: utf-8 -*-

import numpy as np

import utils


def random_distribution(n, n_units):
    p = np.random.uniform(size=2 ** n_units)
    return from_distribution(n, 1. / np.sum(p) * p)


def from_distribution(n, p):
    assert(abs(np.sum(p) - 1.) < 1e-12), 'Distribution p must be normalized.'
    assert(len(p) % 2 == 0), 'Not a distribution over binary states.'
    n_units = int(np.log2(len(p)))
    v = np.empty((n, n_units))
    states = np.array(list(utils.all_possible_states(n_units)))
    indices = list(range(2 ** n_units))
    v = states[np.random.choice(indices, size=n, p=p)]
    return v


def lxor(n):
    v = np.random.randint(0, 2, size=(n, 3))
    v[:, 2] = np.logical_xor(v[:, 0], v[:, 1])
    return v


def land(n):
    v = np.random.randint(0, 2, size=(n, 3))
    v[:, 2] = np.logical_and(v[:, 0], v[:, 1])
    return v
