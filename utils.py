# -*- coding: utf-8 -*-

import collections
import itertools
import numpy as np


def DKL(p, q):
    """returns the Kullback-Leibler divergence of distributions p and q

    """
    assert(np.sum(p) - 1. < 1e-12), 'Distributions must be normalized.'
    assert(np.sum(q) - 1. < 1e-12), 'Distributions must be normalized.'
    assert(np.all(p > 0.)), 'Invalid values in distribution.'
    assert(np.all(q > 0.)), 'Invalid values in distribution.'

    return np.sum(p * np.log(p / q))


def state_array_to_string(s):
    return ''.join(np.array(s, dtype=str))


def get_states_as_strings(n):
    """returns all possible states as strings for n binary units"""
    return np.array([state_array_to_string(s) for s in all_possible_states(n)])


def all_possible_states(n):
    return itertools.product([0, 1], repeat=n)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def psigmoid(x):
    return np.array(sigmoid(x) > np.random.rand(len(x)), dtype=int)


def joint_distribution(states, steps_warmup, prior=None):
    steps_tot = len(states[steps_warmup:])
    n = len(states[0])
    state_counter = collections.defaultdict(int)
    if prior is None:
        pass
    elif prior == 'uniform':
        for s in all_possible_states(n):
            state_counter[tuple(s)] = 1
        steps_tot += 2 ** n
    else:
        raise NotImplementedError('Unknown prior.')
    for s in states[steps_warmup:]:
        state_counter[tuple(s)] += 1
    hist = np.zeros(2 ** n)
    for i, s in enumerate(all_possible_states(n)):
        hist[i] = state_counter[tuple(s)]

    return 1. * hist / np.sum(hist)
