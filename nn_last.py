from pprint import pprint
import pickle as pcl
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_path = './data.csv'


def main():
    df = pd.read_csv(df_path, header=None)
    X = df.loc[:, df.columns != 1].to_numpy()
    Y = df[1].to_numpy()
    Y = np.where((Y == 'M'), 1, 0)
    X = feature_scale(X, mode='standart')
    nn = NN()
    nn.fit(X, Y, plot=False)
    # nn.fit(X, Y, plot=True)


class NN:

    def __init__(self, layers=2, early_stopping=True):
        self.epoches = 0
        self.A = []
        self.weights = []
        self.layers = layers  # number hidden layers
        self.es = True
        self.bias = []
        self.cols = 0  # columns in dataset
        self._activ = np.vectorize(activ)
        self.deltas = []
        self.m = None  # number of examples
        self.d = None
        self.lamb = 0.1
        self.lr = 0.05
        self._batch_size = 5
        self.prev_epoch_weights = None
        self.val_loses = []
        self.loses = []

    def predict(self, X_test):
        res = []
        for xi in X_test:
            self._forward(xi)
            res.append(self._softmax(self.A[-1]))
        return res

    def fit(self, X, Y, epoches=5, batch_size=5, plot=False, save=True):  # add option to calc and print validation metrics
        np.random.seed(42)
        self.m, self.cols = X.shape
        self._batch_size = batch_size
        self.epoches = epoches
        self._init_weights()
        self._init_bias()
        split_index = round(X.shape[0] * 0.8) + 1
        X_train, X_test = X[:split_index], X[split_index:]
        Y_train, Y_test = Y[:split_index], Y[split_index:]
        start_idx, end_idx = 0, 0
        last_epoch = 0
        for epoch in range(self.epoches):
            self.prev_epoch_weights = self.weights
            for _ in range(X.shape[0] // batch_size):
                self._init_d()
                X_batch = X_train[start_idx:end_idx + batch_size]
                Y_batch = Y_train[start_idx:end_idx + batch_size]
                for xi, yi in zip(X_batch, Y_batch):
                    self._forward(xi)
                    self._backprop(yi)
                self._update_weights()
                start_idx += batch_size
                end_idx += batch_size
            loss = self._loss(X_train, Y_train)
            self.loses.append(loss)
            val_lose = self._loss(X_test, Y_test)
            print(f"epoch {epoch + 1}/{epoches} - loss: {round(loss, 4)} - val_loss: {round(val_lose, 4)}")
            last_epoch = epoch
            if self.es and epoch != 0 and self.val_loses[-1] > val_lose:
                self.weights = self.prev_epoch_weights
                print('Early stopping happened.')
                self.val_loses.append(val_lose)
                break
            self.val_loses.append(val_lose)
        acc = self._accuracy(X_test, Y_test)
        print(f'Accuracy: {round(acc * 100, 5)}%')
        prec = self.precision(X_test, Y_test)
        print(f'Precision: {round(prec * 100, 5)}%')
        recall = self.recall(X_test, Y_test)
        print(f'Recall: {round(recall * 100, 5)}%')
        f1_score = self.f1_score(X_test, Y_test)
        print(f'F1 score: {round(f1_score * 100, 5)}%')
        if plot:
            plt.plot(range(last_epoch + 1), self.val_loses)
            plt.xlabel('epoch')
            plt.ylabel('val_lose')
            plt.show()
        if save:
            self.save_weights()

    def f1_score(self, X, Y):
        prec = self.precision(X, Y)
        recall = self.recall(X, Y)
        return (2 * prec * recall) / (prec + recall)

    def recall(self, X, Y):
        preds = self.predict(X)
        correct = 0
        false_negative = 0
        for pred, yi in zip(preds, Y):
            if pred[0] > pred[1]:
                val = 1
            else:
                val = 0
            if val == yi:
                correct += 1
            if val == 0 and yi == 1:
                false_negative += 1
        return correct / (correct + false_negative)

    def precision(self, X, Y):
        preds = self.predict(X)
        correct = 0
        false_positive = 0
        for pred, yi in zip(preds, Y):
            if pred[0] > pred[1]:
                val = 1
            else:
                val = 0
            if val == yi:
                correct += 1
            elif val == 1:
                false_positive += 1
        return correct / (correct + false_positive)

    def _loss(self, X, Y):
        m = X.shape[0]
        sum1 = 0
        for xi, yi in zip(X, Y):
            if yi == 1:
                yi_vec = np.array([1, 0])
            else:
                yi_vec = np.array([0, 1])
            self._forward(xi)
            sum2 = 0
            for xi_k, yi_k in zip(self.A[-1], yi_vec):
                if yi_k == 1:
                    sum2 += yi_k * np.log(xi_k)
                else:
                    sum2 += (1 - yi_k) - np.log(1 - xi_k)
            sum1 += sum2
        res = -sum1 / m
        for weight in self.weights:
            temp = weight ** 2
            res += temp.sum()
        return res

    def _accuracy(self, X, Y):
        preds = self.predict(X)
        correct = 0
        for pred, yi in zip(preds, Y):
            if pred[0] > pred[1]:
                val = 1
            else:
                val = 0
            if val == yi:
                correct += 1
        N = Y.shape[0]
        return correct / N

    def _cross_entropy(self, X, Y):
        preds = self.predict(X)
        error = 0
        for pred, yi in zip(preds, Y):
            if yi == 1:
                error += np.array([1, 0]) @ np.log(pred)
            else:
                error += (1 - np.array([0, 1])) @ np.log(1 - pred)
        N = Y.shape[0]
        return -error / N

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

    def load_weights(self):
        try:
            file = open('./weights.pcl', mode='rb').read()
        except FileNotFoundError:
            print('There is no file with weights.(must be "./weights.pcl")')
            sys.exit()
        weights = pcl.loads(file)
        self.weights = weights['weights']
        self.bias = weights['bias']

    def save_weights(self):
        weights = {'weights': self.weights, 'bias': self.bias}
        file = open('./weights.pcl', mode='bw+')
        pickle_string = pcl.dumps(weights)
        file.write(pickle_string)
        file.close()


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
