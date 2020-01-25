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
    nn.fit(X, Y)

class NeuralNetwork:
    def __init__(self, hidden_layers=2, epoches=100):
        self.hidden_layers = hidden_layers
        self.epoches = epoches
        self.weights = []
        self.deltas_big = []
        self.deltas_small = []
        self._activ = np.vectorize(activ)
        self.A = []

    def fit(self, X, Y):
        self._init_weights(X.shape[1])
        for epoch in range(self.epoches):
            self._init_delta_big(X.shape[1])
            for xi, yi in zip(X, Y):
                self._forward(xi)
                delta_small_last = self._calc_delta_small_last(yi)
                self._calc_deltas_small(delta_small_last)
                break
            break

    def _calc_deltas_small(self, delta_small_last):
        self.deltas_small = [delta_small_last]
        rev_weights = self.weights[::-1]
        rev_A = self.A[1:]
        rev_A = rev_A[::-1]
        for weight, ai, _ in zip(rev_weights, rev_A, range(self.hidden_layers)):
            next_delta_small = self.deltas_small[0]
            temp1 = weight @ next_delta_small
            temp2 = ai * (1 - ai)
            res = temp1 * temp2.T
            self.deltas_small.append(res)


    def _calc_delta_small_last(self, yi):
        if yi == 1:
            yi_vec = np.array([1, 0])[np.newaxis].T
        else:
            yi_vec = np.array([0, 1])[np.newaxis].T
        return self.A[-1] - yi_vec

    def _forward(self, xi):
        a1 = np.insert(xi, 0, 1)  # inserting bias
        self.A = [a1[np.newaxis].T]
        for weight, _ in zip(self.weights, range(self.hidden_layers)):
            prev_a = self.A[-1]
            ai = weight.T @ prev_a
            ai = self._activ(ai)
            ai = np.insert(ai, 0, 1)
            ai = ai[np.newaxis].T
            self.A.append(ai)
            # break
        last_a = self.weights[-1].T @ self.A[-1]
        last_a = self._softmax(last_a)
        self.A.append(last_a)
        print("A shapes:")
        for _ in self.A:
            print(_.shape)
        print('-------------------')

    def _init_delta_big(self, rows):
        for _ in range(self.hidden_layers):
            delta_big = np.zeros((rows + 1, rows))
            self.deltas_big.append(delta_big)
        delta_big_last = np.zeros((rows + 1, 2))
        self.deltas_big.append(delta_big_last)

    def _init_weights(self, rows):
        for _ in range(self.hidden_layers):
            weight = np.random.rand(rows + 1, rows)
            self.weights.append(weight)
        last_weight = np.random.rand(rows + 1, 2)
        self.weights.append(last_weight)
        print('Weight shapes:')
        for _ in self.weights:
            print(_.shape)
        print('-----------------')

    def _softmax(self, x):
        b = x.max()  # trick for numerical stability https://bit.ly/36lGDaN
        y = np.exp(x - b)
        return y / y.sum()


def feature_scale(X, mode='standart'):
    if mode == 'standart':
        return (X - X.mean(axis=0)) / X.std(axis=0)
    if mode == "minmax":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    if mode == 'mean':
        return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def activ(x, mode='relu'):
    # add another activation functions
    if mode == 'sigmoid':
        return 1 if x >= 0.5 else 0
    if mode == 'relu':
        return x if x > 0 else 0


if __name__ == '__main__':
    main()