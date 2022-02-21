from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy.optimize import brentq
import os
import json
from theory import LikelihoodEstimator
import sys
from multiprocessing import Pool as ThreadPool

N = 1000
n_threads = 8
n_batch = 10


alphas = np.concatenate((np.array([0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 25)))) #25
gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 50))

scales = [.1, .3, .5, .7, 0.8, 0.9]

for scale in scales:
    params = {'scale': scale, 'sparsity': 1.}

    def run_one_seed(seed, pars):
        scale = pars['scale']
        model_to_fit = CenteredGM(N, seed=seed, precision=pars, silent=False)
        sys.stdout.flush()
        explorer = LikelihoodEstimator(model_to_fit, n_batches=n_batch, name='sparsity_one/' + 'J_{:.2e}/seed{}_N_'.format(scale, seed) + '{}')
        np.save('out/sparsity_one/J_{:.2e}/seed{}_N_{}/precision.npy'.format(scale, seed, N), model_to_fit.precision)
        explorer.grid_exploration(alphas, gammas, verbosity=1, find_cross=True)

    pool = ThreadPool(n_threads)

    _ = pool.starmap(run_one_seed, zip(range(n_threads), [params for _ in range(n_threads)]))
