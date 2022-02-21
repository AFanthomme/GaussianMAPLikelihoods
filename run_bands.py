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



alphas = np.concatenate((np.array([0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 25)))) #25
gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 50))

# scales = [.3, .7, ]
scales = [.1]
widths = [1, 5, 10, 20, 40, 100, 300, 500]

for coupling in scales:
    for width in widths:
        params = {'scale': coupling, 'bandwidth': width, 'rescale': True}

        def run_one_seed(seed, params):
            coupling = params['scale']
            width = params['bandwidth']
            np.random.seed(seed)
            model_to_fit = CenteredGM(N, precision=params, silent=False)
            sys.stdout.flush()
            explorer = LikelihoodEstimator(model_to_fit, n_batches=10, name='bands/' + 'J_{:.2e}_width_{}/seed{}_N_'.format(coupling, width, seed) + '{}')
            np.save('out/bands/J_{:.2e}_width_{}/seed{}_N_{}/precision.npy'.format(coupling, width, seed, N), model_to_fit.precision)
            explorer.grid_exploration(alphas, gammas, verbosity=1)

        pool = ThreadPool(n_threads)

        _ = pool.starmap(run_one_seed, zip(range(n_threads), [params for _ in range(n_threads)]))
