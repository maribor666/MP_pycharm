import argparse

import pandas as pd
import numpy as np

import NN

df_path = './data.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', default=df_path, metavar='df', help='Dataframe path.')
    parser.add_argument('--metrics', default=True, help='Turn on metrics.')
    args = parser.parse_args()

    df = pd.read_csv(args.df, header=None)
    X = df.loc[:, df.columns != 1].to_numpy()
    Y = df[1].to_numpy()
    Y = np.where((Y == 'M'), 1, 0)
    X = NN.feature_scale(X, mode='standart')
    split_index = round(X.shape[0] * 0.8) + 1
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    nn = NN.NN()
    nn.load_weights()
    nn.predict(X_test)
    if args.metrics:
        nn.calc_metrics(X_test, Y_test)


if __name__ == '__main__':
    main()
