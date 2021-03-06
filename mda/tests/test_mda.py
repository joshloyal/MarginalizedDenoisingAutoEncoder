import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

import mda


file_path = os.path.dirname(__file__)
fixture_path = os.path.join(file_path, 'weights.npy')
data_path = os.path.join(file_path, 'kin8nm_train.csv')


@pytest.mark.thisone
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

    mDA = mda.SMDAutoencoder(n_layers=4, noise_level=0.5)
    h = mDA.fit_transform(X)


def test_mda_pipeline():
    df = pd.read_csv(data_path, header=None)
    y = df.pop(8).values
    X = df.values

    #mDA = mda.MarginalizedDenoisingAutoencoder(noise_level=0.5)
    mDA = mda.SMDAutoencoder(n_layers=4, noise_level=0.5)
    X_mda = mDA.fit_transform(X)

    print('MDA features')
    X_train, X_test, y_train, y_test = train_test_split(X_mda, y, test_size=0.2, random_state=2)
    est = RandomForestRegressor(n_estimators=100, random_state=123)
    est.fit(X_train, y_train)
    print(mean_squared_error(y_test, est.predict(X_test)))


    print('No MDA')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    est = RandomForestRegressor(n_estimators=100, random_state=123)
    est.fit(X_train, y_train)
    print(mean_squared_error(y_test, est.predict(X_test)))
