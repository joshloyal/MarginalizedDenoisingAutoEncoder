import os

import numpy as np
import pandas as pd


import mda


file_path = os.path.dirname(__file__)
fixture_path = os.path.join(file_path, 'weights.npy')
data_path = os.path.join(file_path, 'kin8nm_train.csv')


def test_mda_matlab():
    df = pd.read_csv(data_path, header=None)
    df.pop(8)
    X = df.values

    mDA = mda.MarginalizedDenoisingAutoencoder(noise_level=0.5)
    h = mDA.fit_transform(X)

    W_test = np.loadtxt(fixture_path)

    #print(mDA.weights[0, :])
    #print(mDA.biases[-1])
    #print
    #print(W_test[0, :])

def test_smda_matlab():
    df = pd.read_csv(data_path, header=None)
    df.pop(8)
    X = df.values

    mDA = mda.StackedMarginalizedDenoisingAutoencoder(n_layers=4, noise_level=0.5)
    h = mDA.fit_transform(X)
    print(h)
