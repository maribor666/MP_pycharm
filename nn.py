from pprint import pprint

import numpy as np
import pandas as pd
df_path = './data.csv'


def main():
    df = pd.read_csv(df_path, header=None)
    X = df.loc[:, df.columns != 1].to_numpy()
    Y = df[1].to_numpy()
    Y = np.where((Y == 'M'), 1, 0)
    X = feature_scale(X, mode='standart')
    # divide on train and test dataset
    # mb use cross-validation
    np.random.seed(1)
    nn = NeuralNetwork(epoches=3)
    distributions = nn.fit(X, Y)


class NeuralNetwork:
    def __init__(self, hidden_layers=2, epoches=100, activ_type='sigmoid'):
        self.hidden_layers = hidden_layers
        self.epoches = epoches
        self.epoch = None
        self.A = []
        self.weights = []
        self.m = 0
        self.deltas_big = []
        self.deltas_small = []
        self.history = []
        self.activ_type = activ_type  # change to ReLu
        self._activ = np.vectorize(activ)
        self.lam = 0.01  # coefficient for regularization
        self.lr = 0.05  # learning rate

    def _init_weights(self, cols):
        for _ in range(self.hidden_layers):
            theta = np.random.rand(cols + 1, cols)
            self.weights.append(theta)
        self.weights.append(np.random.rand(cols + 1, 2))

    def _init_deltas_big(self, cols):
        for _ in range(self.hidden_layers):
            delta_big_i = np.zeros((cols + 1, cols))
            self.deltas_big.append(delta_big_i)
        self.deltas_big.append(np.zeros((cols + 1, 2)))

    def _forward(self, xi):
        a1 = np.insert(xi, 0, 1)
        a1 = a1[np.newaxis].T
        self.A = [a1]
        for weight, _ in zip(self.weights, range(self.hidden_layers)):
            prev_a = self.A[-1]
            next_a = weight.T @ prev_a
            next_a = np.insert(next_a, 0, 1)[np.newaxis].T  # inserting bias(bias = 1)
            next_a = self._activ(next_a)
            self.A.append(next_a)
        last_a = self.weights[-1].T @ self.A[-1]
        self.A.append(last_a)

    def _backprop(self, yi):
        if yi == 1:
            yi_vec = np.array([1, 0])[np.newaxis].T
        else:
            yi_vec = np.array([0, 1])[np.newaxis].T
        last_a_softmax = self._softmax(self.A[-1])
        last_delta_small = last_a_softmax - yi_vec
        self.deltas_small = [last_delta_small]
        rev_weights = self.weights[::-1]
        rev_A = self.A[1::-1]
        for weight, ai, _ in zip(rev_weights, rev_A, range(self.hidden_layers)):
            next_delta_small = self.deltas_small[0]
            if _ == 0:
                temp1 = weight @ next_delta_small
            else:
                temp1 = weight.T @ next_delta_small
                temp1 = np.insert(temp1, 0, 1)[np.newaxis].T
            temp2 = ai * (1 - ai)
            delta_small = temp1 * temp2
            self.deltas_small.insert(0, delta_small)

    def _update_weights(self):
        A = self.A[:-1]
        new_big_deltas = []
        for ai, delta_small_i, delta_big_i, _ in zip(A, self.deltas_small, self.deltas_big, range(self.hidden_layers + 1)):
            res = delta_small_i @ ai.T
            if _ != self.hidden_layers:
                res = np.delete(res, 0, axis=1)
            else:
                res = res.T
            new_big_delta = delta_big_i + res
            new_big_deltas.append(new_big_delta)
        self.deltas_big = new_big_deltas
        new_weights = []
        for weight, delta_big_i in zip(self.weights, self.deltas_big):
            D = delta_big_i / self.m + self.lam * weight  # regularization here
            new_weight = weight + self.lr * D
            new_weights.append(new_weight)
        self.weights = new_weights

    def fit(self, X, Y):
        self._init_weights(X.shape[1])
        self._init_deltas_big(X.shape[1])
        self.m = X.shape[0]
        for epoch in range(self.epoches):
            self.epoch = epoch
            for xi, yi in zip(X, Y):
                self._forward(xi)
                # store self.A[-1] only for last epoch
                self._backprop(yi)
                self._update_weights()
        # return softmaxes for all A[-1] values. save A[-1] in self.history or etc
        return 1

    def _softmax(self, x):
        b = x.max()  # trick for numerical stability https://bit.ly/36lGDaN
        y = np.exp(x - b)
        return y / y.sum()


def activ(val, mode='sigmoid'):
    # add another activation functions
    if mode == 'sigmoid':
        return 1 if val >= 0.5 else 0


def feature_scale(X, mode='standart'):
    if mode == 'standart':
        return (X - X.mean(axis=0)) / X.std(axis=0)
    if mode == "minmax":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    if mode == 'mean':
        return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))


if __name__ == '__main__':
    main()
