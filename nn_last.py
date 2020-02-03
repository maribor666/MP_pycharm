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
    # split to train and test
    split_index = round(X.shape[0] * 0.8) + 1
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    np.random.seed(42)
    nn = NN()
    nn.fit(X_train, Y_train)
    preds = nn.predict(X_test)
    correct = 0
    b_num = 0
    for pred, yi in zip(preds, Y_test):
        if pred[0] > pred[1]:
            val = 1
        else:
            val = 0
            b_num += 1
        if val == yi:
            correct += 1
    N = Y_test.shape[0]
    acc = correct / N
    print(acc)
    error = 0
    for pred, yi in zip(preds, Y_test):
        if yi == 1:
            error += np.array([1, 0]) @ np.log(pred)
        else:
            error += (1 - np.array([0, 1])) @ np.log(1 - pred)
    cross_entropy = -error / N
    print('cross-ectropy error:', cross_entropy)


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
        self.lamb = 0.1
        self.lr = 0.05
        self.store = False
        self.history = []
        self.d = []
        self._batch_size = 5

    def predict(self, X_test):
        res = []
        for xi in X_test:
            self._forward(xi)
            res.append(self._softmax(self.A[-1]))
        return res

    def fit(self, X, Y, epoches=1, batch_size=5):
        self.m, self.cols = X.shape
        self._batch_size = batch_size
        self.epoches = epoches
        self._init_weights()
        self._init_bias()
        start_idx, end_idx = 0, 0
        for epoch in range(self.epoches):
            if epoch == self.epoches - 1:
                self.store = True
            for _ in range(X.shape[0] // batch_size):
                self._init_d()
                # try batch
                X_batch = X[start_idx:end_idx + batch_size]
                Y_batch = Y[start_idx:end_idx + batch_size]
                for xi, yi in zip(X_batch, Y_batch):
                    self._forward(xi)
                    self._backprop(yi)
                    if self.store:
                        self.history.append(self._softmax(self.A[-1]))
                    # break
                self._update_weights()
                start_idx += batch_size
                end_idx += batch_size
            # break

    def _update_weights(self):
        new_weights = []
        for di, weight in zip(self.d, self.weights):
            new_weight = weight - self.lr * (di / self._batch_size + self.lamb * weight)
            new_weights.append(new_weight)
        self.weights = new_weights

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
        for weight, ai in zip(rev_weights, rev_A):
            next_delta = self.deltas[0]
            temp1 = weight @ next_delta
            temp2 = ai * (1 - ai)
            res = temp1 * temp2
            self.deltas.insert(0, res)

        for ai, delta_i, _ in zip(self.A, self.deltas[1:], range(len(self.d))):
            res = delta_i[np.newaxis].T @ ai[np.newaxis]
            self.d[_] += res.T

    def _forward(self, xi):
        self.A = [xi]
        for weight, bias_i in zip(self.weights, self.bias):
            prev_a = self.A[-1]
            res = prev_a @ weight + bias_i
            res = self._activ(res)
            self.A.append(res)

    def _init_bias(self):
        # self.bias = np.random.randn(self.cols)
        self.bias = np.zeros(self.cols)

    def _init_weights(self):
        weights = []
        for _ in range(self.layers):
            weight = np.random.randn(self.cols, self.cols)
            weights.append(weight)
        last_weight = np.random.randn(self.cols, 2)
        weights.append(last_weight)
        self.weights = weights

    def _init_d(self):
        d = []
        for weight in self.weights:
            di = np.zeros(weight.shape)
            d.append(di)
        self.d = d

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
    if mode == 'relu':
        return x if x > 0 else 0


if __name__ == '__main__':
    main()
