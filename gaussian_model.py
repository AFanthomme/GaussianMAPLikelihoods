import numpy as np
from scipy.optimize import brentq

def generate_random_matrix(dim, seed=0, scale=1., sparsity=1., rescale=False, bandwidth=None):
    rng = np.random.RandomState(seed)
    n_coeffs = dim ** 2
    n_zeros = (1.-sparsity) * n_coeffs

    sigma = rng.normal(np.zeros([dim,dim], dtype=np.float64), scale/np.sqrt(dim))
    sigma = (sigma + sigma.T) / np.sqrt(2)

    if sparsity < 1:
        mask = np.random.choice([1., 0.], size=(dim, dim), p=[sparsity, 1-sparsity])
        for i in range(dim):
            for j in range(i):
                sigma[i,j] *= mask[i,j]
                sigma[j,i] *= mask[i,j]
        # Could multiply here to ensure same 'eig support length' no matter sparsity
        if rescale:
            sigma /= np.sqrt(sparsity)

    elif bandwidth is not None:
        for i in range(dim):
            for j in range(i):
                # if np.abs(i-j) > bandwidth:
                if i-j > bandwidth and dim-(i-j) > bandwidth:
                    sigma[i,j] *= 0
                    sigma[j,i] *= 0

        if rescale:
            sigma /= np.sqrt(np.min([2*bandwidth/dim, 1.]))

    return sigma - np.diag(np.diag(sigma))

def normalize(precision):
    # Take a null-diagonal J precision matrix and return the same with correct diagonal
    dim = precision.shape[0]
    assert np.all(np.diag(precision) == 0)
    assert np.all(precision == precision.T)

    spectrum = np.real(np.linalg.eig(precision)[0])
    l_max = np.max(spectrum)

    # This is exactly the condition that Tr[(mu-j)^-1] == N
    def surr(x):
        return np.mean(1./(x-spectrum)) - 1.

    # Simple analysis (see notes) guarantees existence of a single root above l_max
    # It must be before l_max+1, and using this one guarantees mu-j pd with correct trace
    mu = brentq(surr, l_max + 1e-6, l_max + 1.05)


    return np.diag(mu * np.ones(dim)) - precision


class CenteredGM:
    def __init__(self, dim, seed=0, precision=None, silent=False):
        self.dim = dim

        np.random.seed(seed)

        if isinstance(precision, np.ndarray):
            self.precision = normalize(precision)
        elif isinstance(precision, dict):
            self.precision = normalize(generate_random_matrix(self.dim, **precision, seed=seed))

        # Sanity checks before model build
        assert self.precision.shape == (self.dim, self.dim)
        assert np.mean(np.diag(np.linalg.inv(self.precision))) - 1. < 1e-3
        assert np.all(np.linalg.eigh(np.linalg.inv(self.precision))[0]) - 1. < 1e-3

        self.covariance = np.linalg.inv(self.precision)

        try:
            assert np.sum(np.diag(self.covariance)) - self.dim < 1e-2
        except AssertionError:
            print('Error, mean eig of C : {}'.format(np.mean(np.diag(self.covariance))))

        if not silent:
            print('Mean eig of C : {}'.format(np.mean(np.diag(self.covariance))))
            print("precision matrix : ", self.precision, '\n\n\n\n\n')
            print("covariance matrix : ", self.covariance)

    def sample(self, n_samples):
        # Draw from multivariate normal of the right covariance
        tmp = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.covariance, size=n_samples)
        assert tmp.shape == (n_samples, self.dim)
        return tmp

    def get_empirical_C(self, n_samples, return_samples=False):
        n_samples = int(n_samples)
        if n_samples == 0:
            n_samples = 1
        obs = self.sample(n_samples)
        assert obs.shape == (n_samples, self.dim)
        assert np.dot(obs.T, obs).shape == (self.dim, self.dim)
        C = np.dot(obs.T, obs) / n_samples
        C *= self.dim / np.sum(np.diag(C))

        if return_samples:
            return C, obs
        else:
            return C
