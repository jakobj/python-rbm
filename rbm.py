import numpy as np

import utils


class RBM(object):
    def __init__(self, n_visible, n_hidden, beta=1.):
        self._beta = beta

        self._n_visible = n_visible
        self._n_hidden = n_hidden

        self._b_visible = 0.5 * (np.random.rand(n_visible) - 0.5)
        self._b_hidden = 0.5 * (np.random.rand(n_hidden) - 0.5)
        self._w = 0.5 * (np.random.beta(2, 2, size=(n_hidden, n_visible)) - 0.5)

    def _markov_step_vhv(self, visible):
        hidden_sample = self._sample_h_given_v(visible)
        visible_sample = self._sample_v_given_h(hidden_sample)
        return visible_sample, hidden_sample

    def _markov_step_hvh(self, hidden):
        visible_sample = self._sample_v_given_h(hidden)
        hidden_sample = self._sample_h_given_v(visible_sample)
        return visible_sample, hidden_sample

    def _sample_v_given_h(self, hidden):
        return utils.psigmoid(self._beta * (np.dot(self._w.T, hidden) + self._b_visible))

    def _sample_h_given_v(self, visible):
        return utils.psigmoid(self._beta * (np.dot(self._w, visible) + self._b_hidden))

    def _train_batch(self, inputs, learning_rate, k):
        assert(k == 1)  # TODO (support for CD-k)

        dw = 0
        dbv = 0
        dbh = 0

        for s in inputs:
            v0 = s.copy()
            h0 = self._sample_h_given_v(v0)

            vk = self._sample_v_given_h(h0)
            phk = utils.sigmoid(self._beta * (np.dot(self._w, vk) + self._b_hidden))

            dw += (np.outer(h0, v0) - np.outer(phk, vk))
            dbv += (v0 - vk)
            dbh += (h0 - phk)

            # ph0 = sigmoid(self._beta * (np.dot(self._w, v0) + self._b_hidden))

            # pvk = sigmoid(self._beta * (np.dot(self._w.T, h0) + self._b_visible))
            # vk = self._sample_v_given_h(h0)

            # phk = sigmoid(self._beta * (np.dot(self._w, vk) + self._b_hidden))
            # hk = self._sample_h_given_v(vk)

            # dw += np.outer(ph0, v0) - np.outer(phk, pvk)
            # dbv += v0 - pvk
            # dbh += ph0 - phk

            # vk = s.copy()
            # for _ in range(k):
            #     vk, hk = self._markov_step_vhv(vk)

            # dw += np.outer(h0, v0) - np.outer(hk, vk)
            # dbv += v0 - vk
            # dbh += h0 - hk

        self._w += 1. / len(inputs) * learning_rate * dw
        self._b_visible += 1. / len(inputs) * learning_rate * dbv
        self._b_hidden += 1. / len(inputs) * learning_rate * dbh

    def fit(self, data, learning_rate, batch_size=1, k=1, sampling_interval=None):
        error = []
        data_dist = utils.joint_distribution(data, 0, prior='uniform')

        for i in range(int(len(data) / batch_size)):
            self._train_batch(data[i:i + batch_size], learning_rate, k)

            if sampling_interval is not None and i > 0 and i % sampling_interval == 0:
                dist = utils.joint_distribution(self.sample(200, 10), 0, prior='uniform')
                error.append(utils.DKL(data_dist, dist))

        return error

    def sample(self, n, trials=1):
        states = []

        for _ in range(trials):
            visible = np.random.randint(0, 2, self._n_visible)
            for _ in range(n):
                visible, hidden = self._markov_step_vhv(visible)
                states.append(visible.copy())
        return np.array(states)
