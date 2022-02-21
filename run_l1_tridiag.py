from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy.optimize import brentq
import os
import json
from l1_regularization import L1_regularizedLikelihoodEstimator
import sys
from multiprocessing import Pool as ThreadPool

N = 200
n_threads = 4

alphas = np.concatenate((np.array([0.05, .1, .5]), np.exp(np.linspace(*np.log([0.1, 100]), 50)))) #25


gammas = np.exp(np.linspace(*np.log([1e-3, 5e1]), 100))


scales = [0.2, 0.4]

for coupling in scales:
    sigma = np.zeros((N, N))
    sigma[0, N - 1] = coupling
    sigma[N - 1, 0] = coupling
    for i in range(N - 1):
        sigma[i, i + 1] = coupling
        sigma[i + 1, i] = coupling

    def run_one_seed(seed, precision):
        np.random.seed(seed)
        model_to_fit = CenteredGM(N, precision=precision, silent=False)
        sys.stdout.flush()
        explorer = L1_regularizedLikelihoodEstimator(model_to_fit, n_batches=5, name='tridiag/' + 'J_{:.2e}/seed{}_N_'.format(coupling, seed) + '{}')
        np.save('out_L1/tridiag/J_{:.2e}/seed{}_N_{}/precision.npy'.format(coupling, seed, N), model_to_fit.precision)
        explorer.grid_exploration(alphas, gammas, verbosity=1)

    pool = ThreadPool(n_threads)

    _ = pool.starmap(run_one_seed, zip(range(n_threads), [sigma for _ in range(n_threads)]))
