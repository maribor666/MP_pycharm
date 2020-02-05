import argparse

import pandas as pd
import numpy as np

import NN

df_path = './data.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', default=df_path, metavar='df', help='Dataframe path.')
    parser.add_argument('--layers', default=2, help='Number of hidden layers.')
    parser.add_argument('-es', default=True, help='Turn off early stopping.')
    parser.add_argument('--plot', default=False, help='Turn on learning curse plotting.')
    args = parser.parse_args()

    df = pd.read_csv(args.df, header=None)
    X = df.loc[:, df.columns != 1].to_numpy()
    Y = df[1].to_numpy()
    Y = np.where((Y == 'M'), 1, 0)
    X = NN.feature_scale(X, mode='standart')
    nn = NN.NN(layers=args.layers, early_stopping=args.es)
    nn.fit(X, Y, plot=args.plot)


if __name__ == '__main__':
    main()
