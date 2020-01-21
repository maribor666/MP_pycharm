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
    nn = NeuralNetwork(epoches=1)
    distributions = nn.fit(X, Y)

class NeuralNetwork():
    def __init__(self, hidden_layers=2, epoches=100, activ_type='sigmoid'):
        self.hidden_layers = hidden_layers
        self.epoches = epoches
        self.A = []
        self.weights = []
        self.m = 0
        self.deltas = []
        self.history = []
        self.activ_type = activ_type
        self._activ = np.vectorize(activ)

    def _init_weights(self, cols):
        for _ in range(self.hidden_layers):
            theta = np.random.rand(cols + 1, cols)
            self.weights.append(theta)
        # weights for output layer
        self.weights.append(np.random.rand(cols + 1, 2))

    def _init_deltas(self):
        for weight in self.weights:
            delta = np.random.randn(*weight.shape)
            self.deltas.append(delta)

    def _forward(self, xi):
        a1 = np.insert(xi, 0, 1)
        self.A.append(a1)
        for _, weight in zip(range(self.hidden_layers), self.weights):
            prev_a = self.A[-1]
            print(prev_a.shape, weight.shape)
            res = prev_a @ weight
            res = np.insert(res, 0, 1)
            res = self._activ(res, mode=self.activ_type)
            self.A.append(res)
        last_a = self.A[-1] @ self.weights[-1]
        self.A.append(last_a)

    def _backprop(self, xi, yi):
        pass

    def _update_weights(self):
        pass

    def fit(self, X, Y):
        self._init_weights(X.shape[1])
        self._init_deltas()
        self.m = X.shape[0]
        for epoch in range(self.epoches):
            for xi, yi in zip(X, Y):
                self._forward(xi)
                # store self.A[-1] only for last epoch
                self._backprop(xi, yi)
                self._update_weights()
                break
        # return softmaxes for all A[-1] values. save A[-1] in self.history or etc
        return self._softmax()

    def _softmax(self):
        return 1


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