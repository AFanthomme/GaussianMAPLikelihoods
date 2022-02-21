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


gammas = np.exp(np.linspace(*np.log([5e-5, 2e0]), 100))

# scales = [.5, .7, .3]
scales = [.7, .3, .1, .8, .9]

for scale in scales:
    # because crash...
    if scale == .7:
        continue
    alphas = alphas[np.where(alphas<50)]

    params = {'scale': scale, 'sparsity': 1.}

    def run_one_seed(seed, pars):
        scale = pars['scale']
        model_to_fit = CenteredGM(N, seed=seed, precision=pars, silent=False)
        sys.stdout.flush()
        # explorer = L1_regularizedLikelihoodEstimator(model_to_fit, n_batches=10, name='sparsity_one/' + 'J_{:.2e}/seed{}_N_'.format(scale, seed) + '{}')
        explorer = L1_regularizedLikelihoodEstimator(model_to_fit, n_batches=3, name='sparsity_one/' + 'J_{:.2e}/seed{}_N_'.format(scale, seed) + '{}')
        np.save('out_L1/sparsity_one/J_{:.2e}/seed{}_N_{}/precision.npy'.format(scale, seed, N), model_to_fit.precision)
        do_crossings = False
        explorer.grid_exploration(alphas, gammas, verbosity=1, find_cross=do_crossings)

    pool = ThreadPool(n_threads)

    _ = pool.starmap(run_one_seed, zip(range(n_threads), [params for _ in range(n_threads)]))
