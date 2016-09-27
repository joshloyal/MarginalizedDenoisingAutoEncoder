import numbers

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_arrays


def mDA_sparse(X, noise_level, W_regularizer):
    """mDA_sparse
    """
    raise NotImplementedError


def mDA_dense(X, noise_level, W_regularizer):
    """mDA_dense.

    Fit a Marginalized Denoising Auto-Encoder for dense training data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    noise_level : float, default: 0.5
        The noise level or corruption probability used to corrupt the input.
        Must be in the range [0, 1].

    W_regularizer : float, default: 1e-5
        The value of the regularization term used when solving the
        convex least squares problem for the hidden weights.

    Returns
    -------
    weights : array-like, shape (n_features, n_features)
        The weights of the encoder.

    biases : array-like, shape (n_features, 1)
        The biases of the encoder.
    """
    n_samples, n_features = X.shape

    # setup noise matrix
    q = np.ones(n_features + 1).reshape(-1, 1)
    q[:-1, :] = (1 - noise_level)

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
    reg = np.eye(n_features + 1) * W_regularizer
    reg[-1, -1] = 0.

    # weights = P * Q^-1 = P * Q^T, since Q is symmetric
    # We can re-write this as
    # Q weights^T = P^T, so weights^T = lstsq(Q, P^T)
    # NOTE: weights^T is the weights in (n_samples, n_features) space.
    weights = np.linalg.lstsq(Q + reg, P.T)[0]
    weights = weights[:-1, :]
    biases = weights[-1, :]

    return weights, biases


class MarginalizedDenoisingAutoencoder(BaseEstimator, TransformerMixin):
    """Marginalized Denoising Auto-encoder.

    An denoising auto-encoder is a simple unsupervised algorithm for
    learning a new representation of the input data, which is especially
    useful for domain adaptation. An auto-encoder works by taking and
    input and mapping it through an encoder to a hidden representation.
    The hidden representation is then mapped back (with a decoder) into
    a reconstruction that attempts to reconstruct the original input.
    A denoising auto-encoder adds noise usually in the form of masking
    certain input features, in order to learn a more robust representation.

    Marginalized Denoising Auto-encoders are a computationally efficient
    version of a denoising auto-encoder, which marginalized over the variable
    noise. This results in an algorithm that is convex with a closed-form
    solution, that is must faster computationally than a standard
    denoising auto-encoder. The mDA algorithm finds the optimal weights
    for a linear transformation to reconstructe the original values.

    Parameters
    ----------
    noise_level : float, default: 0.5
        The noise level or corruption probability used to corrupt the input.
        Must be in the range [0, 1].

    W_regularizer : float, default: 1e-5
        The value of the regularization term used when solving the
        convex least squares problem for the hidden weights.
    """
    def __init__(self, noise_level=0.5, W_regularizer=1e-5):
        self.noise_level = noise_level
        self.W_regularizer = W_regularizer

    def fit(self, X, y=None):
        # Add ability to split data into subsets and
        # reconstruct r most frequent features
        if (not isinstance(self.noise_level, numbers.Number) or
                self.noise_level < 0 or
                self.noise_level > 1):
            raise ValueError('Noise level must be in the range [0, 1]; got '
                             '(noise_level=%r)'
                             % self.noise_level)
        if not isinstance(self.W_regularizer, numbers.Number):
            raise ValueError('Weight regularizor must be a number; got '
                             '(W_regularizer=%r)'
                             % self.W_regularizer)

        X, = check_arrays(X, sparse_format='csr', allow_nans=False)

        if sp.issparse(X):
            self.weights, self.biases = mDA_sparse(
                    X, self.noise_level, self.W_regularizer)
        else:
            self.weights, self.biases = mDA_dense(
                    X, self.noise_level, self.W_regularizer)

        return self

    def transform(self, X):
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise AttributeError('This MarginalizedDenoisingAutoencoder '
                                 'instance is not fitted yet.')

        X, check_arrays(X, sparse_format='csr', allow_nans=False)

        return np.tanh(np.dot(X, self.weights) + self.biases)


class SMDAutoencoder(BaseEstimator, TransformerMixin):
    """SMDAutoencoder.
    """
    def __init__(self, n_layers=4, noise_level=0.5, W_regularizer=1e-5):
        self.n_layers = n_layers
        self.noise_level = noise_level
        self.W_regularizer = W_regularizer
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
                        noise_level=self.noise_level, W_regularizer=self.W_regularizer))
            h = self.mdas[layer].fit_transform(h)

        return self

    def transform(self, X):
        return self._forward(X)
