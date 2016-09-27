import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mda


# Hyper-parameters
n_trees = 300
noise_level = 0.5
n_layers = 5

train = pd.read_csv('./data/kin8nm_train.csv', header=None)
y_train = train.pop(8).values
X_train = train.values

validation = pd.read_csv('./data/kin8nm_validation.csv', header=None)
y_val = validation.pop(8).values
X_val = validation.values


mDA = mda.SMDAutoencoder(n_layers=n_layers, noise_level=noise_level)

t0 = time.time()
Xhat_train = mDA.fit_transform(X_train)
Xhat_val = mDA.transform(X_val)
print('mDA fit_transform time: {:2f}'.format(time.time() - t0))

est = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
est.fit(Xhat_train, y_train)
print('mDA RMSE: {:2f}'.format(mean_squared_error(y_val, est.predict(Xhat_val))))

est = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
est.fit(X_train, y_train)
print('Raw RMSE: {:2f}'.format(mean_squared_error(y_val, est.predict(X_val))))
