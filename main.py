import matplotlib.pyplot as plt
import numpy as np

import benchmarks
import rbm
import utils


np.random.seed(23)

n_train = 15000
n_test = 2000
learning_rate = 0.05
batch_size = 1
k = 1

data_train = benchmarks.lxor(n_train)

myrbm = rbm.RBM(3, 4)
initial_states = myrbm.sample(n_test, 10)

error = myrbm.fit(data_train, learning_rate, batch_size, k, 50)

final_states = myrbm.sample(n_test, 10)

data_dist = utils.joint_distribution(data_train, 0, prior='uniform')
initial_dist = utils.joint_distribution(initial_states, 0, prior='uniform')
final_dist = utils.joint_distribution(final_states, 0, prior='uniform')

print(utils.DKL(data_dist, initial_dist), 'vs', utils.DKL(data_dist, final_dist))

plt.subplot(221)
plt.plot(error)
plt.ylim([0., 1.])
plt.subplot(222)
plt.bar(np.arange(2**3) + 0.0, data_dist, color='k', width=0.2, linewidth=0)
plt.bar(np.arange(2**3) + 0.2, initial_dist, color='b', width=0.2, linewidth=0)
plt.bar(np.arange(2**3) + 0.4, final_dist, color='r', width=0.2, linewidth=0)
plt.gca().set_xticklabels(utils.get_states_as_strings(3))
plt.subplot(223)
plt.pcolormesh(initial_states, cmap='gray')
plt.subplot(224)
plt.pcolormesh(final_states, cmap='gray')
plt.show()
