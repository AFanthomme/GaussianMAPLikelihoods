from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import gstd, chi2
import os
import json
from theory import LikelihoodEstimator
import sys
from multiprocessing import Pool as ThreadPool
from copy import deepcopy
os.makedirs('out/j_star_illustrations/', exist_ok=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# For nice print
float_formatter = "{:.5e}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

max = lambda x,y: x if x>y else y
min = lambda x,y: x if x<y else y
n = 2

import logging


def jstar(mu, c, alpha, gamma):
    return (alpha*c+gamma*mu-np.sqrt((alpha*c-gamma*mu)**2 + 4*gamma*alpha)) / (2*gamma)

def find_mu(c, alpha, gamma):
    def surrogate(mu):
        return 2. - 1. / (mu - jstar(mu, c, alpha, gamma))- 1. / (mu - jstar(mu, 2-c, alpha, gamma))

    mu_range = np.linspace(.95, 30, 100)
    root = brentq(surrogate, 0.95, 1e10, maxiter=100)
    return root


gammas = [.01, .1, 1., 10, 100, 1000]
alphas = [1., 10., 100., 1000.]
c_range = np.linspace(0.2, 1., 10000)


mu_blob = np.zeros((len(alphas), len(gammas), len(c_range)))
j1_blob = np.zeros((len(alphas), len(gammas), len(c_range)))
j2_blob = np.zeros((len(alphas), len(gammas), len(c_range)))

for alpha_idx, alpha in enumerate(alphas):
    for gamma_idx, gamma in enumerate(gammas):
        for c_idx, c in enumerate(c_range):
            mu_star = find_mu(c, alpha, gamma)
            mu_blob[alpha_idx, gamma_idx, c_idx] = mu_star
            j1_blob[alpha_idx, gamma_idx, c_idx] = jstar(mu_star, c, alpha, gamma)
            j2_blob[alpha_idx, gamma_idx, c_idx] = jstar(mu_star, 2.-c, alpha, gamma)

        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        axes[0].plot(c_range, mu_blob[alpha_idx, gamma_idx], label=r'$\mu^*$')
        axes[0].set_xlabel(r'Regularization strength $\gamma$')
        axes[0].set_ylabel(r'Value of lagrange multiplier $\mu^*$')
        axes[1].plot(c_range, j1_blob[alpha_idx, gamma_idx], label=r'$j_1^*$')
        axes[1].plot(c_range, j2_blob[alpha_idx, gamma_idx], label=r'$j_2^*$')
        axes[1].set_xlabel(r'Regularization strength $\gamma$')
        axes[1].set_ylabel(r'Value of MAP eigenvalues')
        plt.savefig('out/j_star_illustrations/alpha_{}_gamma_{}.pdf'.format(alpha, gamma))
