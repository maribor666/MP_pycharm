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
    nn = NeuralNetwork(hidden_layers=2)
    nn.fit(X, Y, epoches=1)
    X_test = X
    Y_predicted = nn.predict(X_test)

    total = len(Y)
    correct_answers = 0
    for yi, yi_pred in zip(Y, Y_predicted):
        # print(yi, yi_pred)
        if yi == yi_pred:
            correct_answers += 1
    print(correct_answers / total)


class NeuralNetwork:
    def __init__(self, hidden_layers=2, epoches=100):
        self.epoches = epoches
        self.weights = []
        self.deltas = []
        self.A = []
        self.before_active = []
        self.hidden_layers = hidden_layers

    def _forward(self, xi):
        a1 = np.insert(xi, 0, 1)  # adding bias
        self.A = [a1]
        for weight, _ in zip(self.weights, range(self.hidden_layers)):
            prev_a = self.A[-1]
            res = np.matmul(prev_a, weight)
            res = np.insert(res, 0, 1)
            self.A.append(res)
        # calc a for output layer
        a_output_layer = np.matmul(self.A[-1], self.weights[-1])
        self.before_active.append(a_output_layer)
        a_output_layer = self._activ(a_output_layer)
        self.A.append(a_output_layer)

    def _backprop(self, yi):
        delta_last = self.A[-1] - yi
        self.deltas.append(delta_last)
        rev_weights = self.weights[::-1]
        for ai, weight, _ in zip(self.A[:-1][::-1], rev_weights, range(self.hidden_layers)):
            prev_delta = self.deltas[0]
            g_z = ai * (1 - ai)
            temp = weight.T * prev_delta
            res = temp * g_z
            self.deltas.insert(0, res)

    def _update_weights(self):
        new_weights = []
        for weight, ai, delta_i, _ in zip(self.weights, self.A, self.deltas, range(self.hidden_layers)):
            new_weight = weight + np.matmul(ai, delta_i.T)
            new_weights.append(new_weight)
        last_new_weight = self.weights[-1] + self.A[-1] * self.deltas[-1]
        new_weights.append(last_new_weight)
        self.weights = new_weights

    def fit(self, X, Y, epoches=100):
        self._init_weight(X.shape[1])

        for epoch in range(epoches):
            for xi, yi in zip(X, Y):
                self._forward(xi)
                self._backprop(yi)
                self._update_weights()

    def predict(self, X):
        predicted_vals = []
        for xi in X:
            self._forward(xi)
            res = self.A[-1]
            predicted_vals.append(res)
        return predicted_vals

    def _activ(self, val):
        return 1 if val >= 0.5 else 0

    def _init_weight(self, rows):
        for _ in range(self.hidden_layers):
            theta = np.random.rand(rows + 1, rows)
            self.weights.append(theta)
        # creating weights for output layer
        self.weights.append(np.random.rand(rows + 1, 1))


# it can be in another file
def feature_scale(X, mode='standart'):
    if mode == 'standart':
        return (X - X.mean(axis=0)) / X.std(axis=0)
    if mode == "minmax":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    if mode == 'mean':
        return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))


if __name__ == '__main__':
    main()
