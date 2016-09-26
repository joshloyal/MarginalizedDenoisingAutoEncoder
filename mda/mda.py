import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MarginalizedDenoisingAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.5, alpha=1e-5):
        self.noise_level = noise_level
        self.alpha = alpha  # regularization level

    def fit(self, X, y=None):
        X = X.astype(np.float32)
        n_samples, n_features = X.shape

        # FIXME: This is a full data copy. Don't do this.
        fit_X = np.hstack((X, np.ones((n_samples, 1), dtype=X.dtype))).T

        # setup noise matrix
        q = np.ones((n_features + 1, 1), dtype=X.dtype)
        q[:-1, :] = (1 - self.noise_level)

        # scatter matrix
        S = np.dot(fit_X, fit_X.T)
        Q = S * np.dot(q, q.T)
        np.fill_diagonal(Q, q * np.diag(S))

        P = S[:-1, :] * np.tile(q.T, (n_features,  1))

        reg = np.eye(n_features + 1, dtype=X.dtype) * self.alpha
        reg[-1, -1] = 0.
        weights = np.linalg.lstsq(Q + reg, P.T)[0]

        self.weights = weights[:-1, :]
        self.biases = weights[-1, :]

        return self

    def transform(self, X):
        return np.tanh(np.dot(X, self.weights) + self.biases)


class StackedMarginalizedDenoisingAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_layers=4, noise_level=0.5, alpha=1e-5):
        self.n_layers = n_layers
        self.noise_level = noise_level
        self.alpha = alpha
        self.mdas = []

    def _forward(self, X):
        h = X
        for mda in self.mdas:
            h = mda.transform(h)
        return h

    def fit(self, X, y=None):
        h = X
        for layer in xrange(self.n_layers):
            self.mdas.append(
                    MarginalizedDenoisingAutoencoder(
                        noise_level=self.noise_level, alpha=self.alpha))
            h = self.mdas[layer].fit_transform(h)

        return self

    def transform(self, X):
        return self._forward(X)
