import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from gaussian_model import CenteredGM
from mpl_toolkits.mplot3d import Axes3D
from itertools import product


N = 1000
p = 1000

def check_for_sparse():
    scales = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.])
    n_reals = 10
    overlaps = np.zeros((len(scales), n_reals, p))

    for idx, scale in enumerate(scales):
        for real in range(n_reals):
            params = {'scale': scale, 'sparsity': 1.}
            model_to_fit = CenteredGM(N, seed=real, precision=params, silent=True)
            C = model_to_fit.covariance
            X = model_to_fit.sample(p)
            # print(X.shape)
            assert X.shape == (p, N)
            overlaps[idx, real] = np.diag(X.dot(C).dot(X.T)) / np.sum(X**2, axis=1) # overlap defined using normalized U

    print(np.min(overlaps, axis=(1,2)), np.max(overlaps, axis=(1,2)))
    overlaps = overlaps.reshape(len(scales), -1) / N #1/N * uCu

    plt.figure()
    plt.errorbar(scales, overlaps.mean(axis=1), yerr=overlaps.std(axis=1), label='Measured')
    plt.plot(scales, (1+scales**2)/ N, label='Naive prediction')
    plt.xlabel('Scale')
    plt.ylabel('Overlap')
    plt.yscale('log')
    plt.legend()
    plt.savefig('out/sparsity_one/overlaps.png')

    o = overlaps.mean(axis=1)
    plt.figure()
    plt.plot(scales, (N*o)/((N+1)*o-1))
    plt.xlabel('Scale')
    plt.ylabel('Optimal gamma for 1 pattern')
    plt.yscale('log')
    plt.legend()
    plt.savefig('out/sparsity_one/pred_gamma1.png')


def check_for_tridiag():
    scales = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2., 5., 10., 20])
    n_reals = 10
    overlaps = np.zeros((len(scales), n_reals, p))

    for idx, scale in enumerate(scales):
        for real in range(n_reals):
            sigma = np.zeros((N, N))
            sigma[0, N - 1] = scale
            sigma[N - 1, 0] = scale
            for i in range(N - 1):
                sigma[i, i + 1] = scale
                sigma[i + 1, i] = scale

            model_to_fit = CenteredGM(N, seed=real, precision=sigma, silent=True)
            C = model_to_fit.covariance
            X = model_to_fit.sample(p)
            # print(X.shape)
            assert X.shape == (p, N)
            overlaps[idx, real] = np.diag(X.dot(C).dot(X.T)) / np.sum(X**2, axis=1) # overlap defined using normalized U

    print(np.min(overlaps, axis=(1,2)), np.max(overlaps, axis=(1,2)))
    overlaps = overlaps.reshape(len(scales), -1) / N #1/N * uCu

    plt.figure()
    plt.errorbar(scales, overlaps.mean(axis=1), yerr=overlaps.std(axis=1), label='Measured')
    plt.plot(scales, (1+scales**2) / N, label='Naive prediction')
    plt.xlabel('Scale')
    plt.ylabel('Overlap')
    plt.yscale('log')
    plt.legend()
    plt.savefig('out/tridiag/overlaps.png')

    o = overlaps.mean(axis=1)
    plt.figure()
    plt.plot(scales, (N*o)/((N+1)*o-1))
    plt.xlabel('Scale')
    plt.ylabel('Optimal gamma for 1 pattern')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig('out/tridiag/pred_gamma1.png')


if __name__ == '__main__':
    # check_for_tridiag()
    check_for_sparse()
