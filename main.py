# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import benchmarks
import rbm
import utils


seed_data = 123
seed_weights_biases = 891
seed_learn = 812

n_visible_units = 3
n_hidden_units = 8
n_train_samples = 280000
n_test_samples = 50000
n_test_trials = 5
learning_rate = 0.1
batch_size = 10
k = 1
sampling_interval = 500
# p = [0.3, 0.1, 0.1, 0.5]

np.random.seed(seed_data)
# data_train = benchmarks.from_distribution(n_train_samples, p)
data_train = benchmarks.random_distribution(n_train_samples, n_visible_units)
# data_train = benchmarks.lxor(n_train_samples)
# data_train = benchmarks.random_distribution(n_visible, n_train)

np.random.seed(seed_weights_biases)
myrbm = rbm.RBM(n_visible_units, n_hidden_units)
initial_states = myrbm.sample(n_test_samples, n_test_trials)

print('weights before', myrbm._w)

np.random.seed(seed_learn)
error = myrbm.fit(data_train, learning_rate, batch_size, k, sampling_interval)

print('weights after', myrbm._w)

final_states = myrbm.sample(n_test_samples, n_test_trials)

data_dist = utils.joint_distribution(data_train, 0, prior='uniform')
initial_dist = utils.joint_distribution(initial_states, 0, prior='uniform')
final_dist = utils.joint_distribution(final_states, 0, prior='uniform')

print(utils.DKL(data_dist, initial_dist), 'vs', utils.DKL(data_dist, final_dist))

plt.subplot(221)
plt.plot(error)
plt.ylim([0., 1.])
plt.subplot(222)
plt.bar(np.arange(2**n_visible_units) + 0.0, data_dist, color='k', width=0.2, linewidth=0)
plt.bar(np.arange(2**n_visible_units) + 0.2, initial_dist, color='b', width=0.2, linewidth=0)
plt.bar(np.arange(2**n_visible_units) + 0.4, final_dist, color='r', width=0.2, linewidth=0)
plt.gca().set_xticklabels(utils.get_states_as_strings(3))
plt.subplot(223)
for i in range(n_hidden_units):
    for j in range(n_visible_units):
        plt.plot(np.array(myrbm._a_dw)[:, i, j])
# plt.pcolormesh(initial_states, cmap='gray')
plt.subplot(224)
# plt.pcolormesh(final_states, cmap='gray')
plt.show()
