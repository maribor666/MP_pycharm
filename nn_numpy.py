from pprint import pprint

import numpy as np
import pandas as pd
df_path = './data.csv'


def main():
    df = pd.read_csv(df_path, header=None)
    X = df.loc[:, df.columns != 1].to_numpy()
    # print(X)
    Y = df[1].to_numpy()
    Y = np.where((Y == 'M'), 1, 0)
    # print(Y)

    # feature preprocess should be
    # here
    np.random.seed(42)
    nn = NeuralNetwork()
    nn.fit(X, Y, epoches=1)


class NeuralNetwork:
    def __init__(self, hidden_layers=2, epoches=100):
        self.epoches = epoches
        self.weights = []
        self.A = []
        self.hidden_layers = hidden_layers

    def _forward(self, xi):
        a1 = np.insert(xi, 0, 1)  # adding bias
        self.A = [a1]
        for weight, _ in zip(self.weights, range(self.hidden_layers)):
            prev_a = self.A[-1]
            res = np.matmul(prev_a, weight)
            res = np.insert(res, 0, 1)
            res = np.where((res >= 0.5), 1, 0)  # values after activation func
            self.A.append(res)
        # calc a for output layer
        a_output_layer = np.matmul(self.A[-1], self.weights[-1])
        a_output_layer = self._activ(a_output_layer)
        self.A.append(a_output_layer)
        pprint(self.A)

    def _backprop(self):
        pass

    def _update_weights(self):
        pass

    def fit(self, X, Y, epoches=100):
        print('shape of X:', X.shape)
        print('shape of Y:', Y.shape)
        self._init_weight(X.shape[1])
        for w in self.weights:
            print(w.shape)

        for epoch in range(epoches):
            for xi, yi in zip(X, Y):
                self._forward(xi)
                break

    @staticmethod
    def _activ(val):
        return 1 if val >= 0.5 else 0

    def _init_weight(self, rows):
        for _ in range(self.hidden_layers):
            theta = np.random.rand(rows + 1, rows)
            self.weights.append(theta)
        # creating weights for output layer
        self.weights.append(np.random.rand(rows + 1, 1))


if __name__ == '__main__':
    main()
