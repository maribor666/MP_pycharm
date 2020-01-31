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
    # split to train ant test
    np.random.seed(42)
    nn = NN()
    nn.fit(X, Y)
    correct = 0
    b_num = 0
    for pred, yi in zip(nn.history, Y):
        if pred[0] > pred[1]:
            val = 1
        else:
            val = 0
            b_num += 1
        if val == yi:
            correct += 1
    acc = correct / Y.shape[0]
    print(acc)
    print(b_num)


class NN:
    def __init__(self, layers=2):
        self.epoches = 0
        self.A = []
        self.weights = []
        self.layers = layers  # number hidden layers
        self.cols = 0  # columns in dataset
        self._activ = np.vectorize(activ)
        self.deltas = []
        self.m = None  # number of examples
        self.d = None
        self.lamb = 0.0001
        self.lr = 0.001
        self.store = False
        self.history = []
        self.d = []

    def fit(self, X, Y, epoches=70):
        self.m, self.cols = X.shape
        self.epoches = epoches
        self._init_weights()
        for epoch in range(self.epoches):
            if epoch == self.epoches - 1:
                self.store = True
            self._init_d()
            for xi, yi in zip(X, Y):
                self._forward(xi)
                self._backprop(yi)
                if self.store:
                    self.history.append(self._softmax(self.A[-1]))
                # break
            self._update_weights()
            # break

    def _update_weights(self):
        new_weights = []
        for di, weight in zip(self.d, self.weights):
            new_weight = weight + self.lr * (di / self.m + self.lamb * weight)
            new_weights.append(new_weight)

    def _backprop(self, yi):
        if yi == 1:
            yi_vec = np.array([1, 0])
        else:
            yi_vec = np.array([0, 1])
        last_delta = self.A[-1] - yi_vec
        # print(last_delta)
        self.deltas = [last_delta]
        rev_weights = self.weights[::-1]
        rev_A = self.A[:-1]
        rev_A = rev_A[::-1]
        for weight, ai, _ in zip(rev_weights, rev_A, range(len(rev_A))):
            next_delta = self.deltas[0]
            # print(_)
            # print(weight)
            # print(next_delta)
            # if _ == 0:
            #     temp1 = weight @ next_delta
            # else:
            #     temp1 = weight.T @ next_delta
            temp1 = weight @ next_delta
            # print(temp1)
            temp2 = ai * (1 - ai)
            res = temp1 * temp2
            # print(res)
            self.deltas.insert(0, res)
            # print('------------')

        for ai, delta_i, _ in zip(self.A, self.deltas[1:], range(len(self.d))):
            res = delta_i[np.newaxis].T @ ai[np.newaxis]
            self.d[_] += res.T

    def _forward(self, xi):
        a_first = np.insert(xi, self.cols, 1)
        self.A = [a_first]
        for weight, _ in zip(self.weights, range(len(self.weights))):
            prev_a = self.A[-1]
            res = prev_a @ weight
            res = self._activ(res)
            self.A.append(res)
        # print('A:')
        # pprint(self.A)
        # print('------------')

    def _init_weights(self):
        weights = []
        for _ in range(self.layers):
            weight = np.random.randn(self.cols + 1, self.cols + 1)
            weights.append(weight)
        last_weight = np.random.randn(self.cols + 1, 2)
        weights.append(last_weight)
        self.weights = weights
        # print('weights')
        # pprint(weights)
        # print('------------')

    def _init_d(self):
        d = []
        for weight in self.weights:
            di = np.zeros(weight.shape)
            d.append(di)
        self.d = d
        # print('d')
        # pprint(self.d)
        # print('------------')

    def _softmax(self, x):
        s = np.exp(x).sum()
        return np.exp(x) / s


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
