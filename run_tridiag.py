from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os
import json
from theory import LikelihoodEstimator
import sys
from multiprocessing import Pool as ThreadPool

N = 1000
n_threads = 8


alphas = np.concatenate((np.array([0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 25)))) #25
gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 50))


scales = [.1, .3, .5, .7, 1., 3., 10.]

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
        explorer = LikelihoodEstimator(model_to_fit, n_batches=10, name='tridiag/' + 'J_{:.2e}/seed{}_N_'.format(coupling, seed) + '{}')
        np.save('out/tridiag/J_{:.2e}/seed{}_N_{}/precision.npy'.format(coupling, seed, N), model_to_fit.precision)
        explorer.grid_exploration(alphas, gammas, verbosity=1)

    pool = ThreadPool(n_threads)

    _ = pool.starmap(run_one_seed, zip(range(n_threads), [sigma for _ in range(n_threads)]))
