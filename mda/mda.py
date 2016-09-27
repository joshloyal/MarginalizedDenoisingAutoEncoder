import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MarginalizedDenoisingAutoencoder(BaseEstimator, TransformerMixin):
    """MarginalizedDenoisingAutoencoder.

    Add ability to split data into subsets and reconstruct r most frequent features.

    Add sparse support.
    """
    def __init__(self, noise_level=0.5, alpha=1e-5):
        self.noise_level = noise_level
        self.alpha = alpha  # regularization level

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # setup noise matrix
        q = np.ones(n_features + 1).reshape(-1, 1)
        q[:-1, :] = (1 - self.noise_level)

        # scatter matrix (includes a bias feature)
        S = np.zeros((n_features + 1, n_features + 1))
        S[:n_features, :n_features] = np.dot(X.T, X)

        # bias part of the scatter matrix
        feature_sum = np.sum(X, axis=0)
        S[-1, :-1] = feature_sum
        S[:-1, -1] = feature_sum
        S[-1, -1] = n_samples

        # Q matrix (n_feature + 1, n_feature + 1) matrix
        Q = S * np.dot(q, q.T)
        np.fill_diagonal(Q, q * np.diag(S))

        # P matrix (n_features, n_features + 1) matrix
        P = S[:-1, :] * np.tile(q.T, (n_features,  1))

        # regularization term
        reg = np.eye(n_features + 1) * self.alpha
        reg[-1, -1] = 0.

        # solve for weights
        weights = np.linalg.lstsq(Q + reg, P.T)[0]
        self.weights = weights[:-1, :]
        self.biases = weights[-1, :]

        return self

    def transform(self, X):
        return np.tanh(np.dot(X, self.weights) + self.biases)


class SMDAutoencoder(BaseEstimator, TransformerMixin):
    """SMDAutoencoder.
    """
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
