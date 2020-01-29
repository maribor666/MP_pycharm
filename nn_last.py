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
    # X = X[:, :3]
    np.random.seed(1)
    nn = NN()
    nn.fit(X, Y)
    # pprint(nn.history)
    print(nn.history[0])
    print(nn.history[-2])
    print(nn.history[-1])

class NN:
    def __init__(self, layers=2):
        self.epoches = 0
        self.A = []
        self.weights = []
        self.layers = layers  # number hidden layers
        self.bias = []
        self.cols = 0  # columns in dataset
        self._activ = np.vectorize(activ)
        self.deltas = []
        self.m = None  # number of examples
        self.d = None
        self.lamb = 0.0001
        self.lr = 0.001
        self.store = False
        self.history = []

    def fit(self, X, Y, epoches=70):
        self.m, self.cols = X.shape
        self.epoches = epoches
        self._init_weights()
        self._init_bias()
        for epoch in range(self.epoches):
            if epoch == self.epoches - 1:
                self.store = True
            for xi, yi in zip(X, Y):
                self._forward(xi)
                self._backprop(yi)
                if self.store:
                    self.history.append(self.A[-1])
                # break
            # break

    def _backprop(self, yi):
        if yi == 1:
            yi_vec = np.array([1, 0])
        else:
            yi_vec = np.array([0, 1])
        last_delta = self.A[-1] - yi_vec
        self.deltas = [last_delta]
        rev_weights = self.weights[::-1]
        rev_A = self.A[:-1]
        rev_A = rev_A[::-1]
        for weight, ai in zip(rev_weights, rev_A):
            next_delta = self.deltas[0]
            temp1 = weight @ next_delta
            temp2 = ai * (1 - ai)
            res = temp1 * temp2
            self.deltas.insert(0, res)
        # print('deltas:')
        # pprint(self.deltas[1:])
        # print('-------------')
        d = []
        for ai, delta_i in zip(self.A, self.deltas[1:]):
            res = delta_i[np.newaxis].T @ ai[np.newaxis]
            d.append(res.T)  # test with transpose and without it!!!!!!
        # print('d:')
        # pprint(d)
        new_weights = []
        for di, weight in zip(d, self.weights):
            new_weight = weight + self.lr * (di / self.m + self.lamb * weight)
            new_weights.append(new_weight)
        # print('new weights')
        # pprint(new_weights)


    def _forward(self, xi):
        self.A = [xi]
        for weight, bias_i in zip(self.weights, self.bias):
            prev_a = self.A[-1]
            res = prev_a @ weight + bias_i
            res = self._activ(res)
            self.A.append(res)
        # print('A:')
        # pprint(self.A)
        # print('-------------')

    def _init_bias(self):
        self.bias = np.zeros(shape=self.cols)
        # print('bias:')
        # pprint(self.bias)
        # print('-------------')

    def _init_weights(self):
        weights = []
        for _ in range(self.layers):
            weight = np.random.randn(self.cols, self.cols)
            weights.append(weight)
        last_weight = np.random.randn(self.cols, 2)
        weights.append(last_weight)
        self.weights = weights
        # print('weights:')
        # pprint(self.weights)
        # print('--------------')


def feature_scale(X, mode='standart'):
    if mode == 'standart':
        return (X - X.mean(axis=0)) / X.std(axis=0)
    if mode == "minmax":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    if mode == 'mean':
        return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def activ(x, mode='sigmoid'):
    if mode == 'sigmoid':
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    main()
