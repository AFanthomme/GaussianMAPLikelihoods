from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')
from scipy.optimize import brentq
from scipy.stats import gstd, chi2
import os
import json
from theory import LikelihoodEstimator
import sys
from multiprocessing import Pool as ThreadPool
from copy import deepcopy
os.makedirs('out/small_alpha/', exist_ok=True)
os.makedirs('out/small_alpha/experiments', exist_ok=True)

# For nice print
float_formatter = "{:.5e}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

max = lambda x,y: x if x>y else y
min = lambda x,y: x if x<y else y
n = 1000

import logging


def do_theory():
    n_seeds = 10
    n_draws = 1000

    n_scales = 40
    scales_below_1 = np.exp(np.linspace(*np.log([1e-2, .99]), n_scales//2))
    scales_above_1 = np.exp(np.linspace(*np.log([1.01, 100]), n_scales//2))
    scales = np.hstack([scales_below_1, scales_above_1])[::-1]

    thetas_rescaled = np.zeros((n_scales, n_seeds, n_draws))
    sample_norms = np.zeros((n_scales, n_seeds, n_draws))
    thetas = np.zeros((n_scales, n_seeds, n_draws))
    norms_parallel = np.zeros((n_scales, n_seeds, n_draws))
    eigs = np.zeros((n_scales, n_seeds))

    for scale_idx, scale in enumerate(tqdm(scales)):
        for seed in range(n_seeds):
            params = {'scale': scale, 'sparsity': 1.}
            model = CenteredGM(n, seed=seed, precision=params, silent=True)

            # This is for the theta prediction
            C = model.covariance
            eigsvals, eigsvects = np.linalg.eigh(C)
            idx = np.argmax(eigsvals)
            max_vect = eigsvects[:, idx]
            eigs[scale_idx, seed] = np.max(eigsvals)

            samples = model.sample(n_draws)
            norms = np.diag(samples.dot(samples.T))
            sample_norms[scale_idx, seed] = norms

            v_max_dot_s = samples.dot(max_vect)
            norms_parallel[scale_idx, seed] = v_max_dot_s**2

            dots = np.diag(samples.dot(C).dot(samples.T)) / (n * n) # one n comes from definition of theta, the other from need to normalize samples
            thetas[scale_idx, seed] = dots
            thetas_rescaled[scale_idx, seed] = dots * n / norms

    np.save('out/small_alpha/scales_theory.npy', scales)
    np.save('out/small_alpha/thetas_theory.npy', thetas)
    np.save('out/small_alpha/eigs_theory.npy', eigs)
    np.save('out/small_alpha/thetas_rescaled_theory.npy', thetas_rescaled)
    np.save('out/small_alpha/norms_theory.npy', sample_norms)
    np.save('out/small_alpha/norms_parallel_theory.npy', norms_parallel)

def do_plots():
    folder = 'out/small_alpha/'
    # folder = 'out/small_alpha_cutoff_at_minus7/'
    scales_theory = np.load(folder+'scales_theory.npy')
    thetas_rescaled = np.load(folder+'thetas_rescaled_theory.npy')
    thetas_theory = np.load(folder+'thetas_theory.npy')
    norms_theory = np.load(folder+'norms_theory.npy')
    max_eigs = np.load(folder+'eigs_theory.npy')
    norms_parallel_theory = np.load(folder+'norms_parallel_theory.npy')

    scales_theory_below_1 = scales_theory[np.where(scales_theory<.9)]
    scales_theory_above_1 = scales_theory[np.where(scales_theory>1.1)]

    os.makedirs(folder+'plots', exist_ok=True)


    thetas_theory_filtered = deepcopy(thetas_theory)
    thetas_theory_filtered[np.where(thetas_rescaled<1./n+1e-7)] = np.nan
    thetas_theory_averaged_over_repeats = np.nanmean(thetas_theory_filtered, axis=-1)


    thetas_filtered = deepcopy(thetas_rescaled)
    thetas_filtered[np.where(thetas_filtered<1./n)] = np.nan
    thetas_rescaled_averaged_over_repeats = np.nanmean(thetas_filtered, axis=-1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    # Now get experiment data
    n_scales = 100
    n_seeds = 10
    n_repeats = 100

    scales_below_1 = np.exp(np.linspace(*np.log([1e-2, .9]), n_scales//2))
    scales_above_1 = np.exp(np.linspace(*np.log([1.1, 100]), n_scales//2))
    scales = np.hstack([scales_below_1, scales_above_1])

    cross_blob = np.zeros((n_seeds, n_scales, n_repeats))
    opt_blob = np.zeros((n_seeds, n_scales, n_repeats))

    for scale_idx, scale in enumerate(scales):
        for seed in range(n_seeds):
                opt_blob[seed, scale_idx] = np.loadtxt(folder+'experiments/J_{:.2e}/seed{}_N_{}/gamma_opt.txt'.format(scale, seed, n))
                cross_blob[seed, scale_idx] = np.loadtxt(folder+'experiments/J_{:.2e}/seed{}_N_{}/gamma_cross.txt'.format(scale, seed, n))

    assert not np.all(opt_blob==cross_blob)


    pred_ferro = ((1-thetas_rescaled)**2)
    pred_ferro[np.where(thetas_rescaled<1./n)] = np.nan
    pred_ferro_mean = np.nanmean(pred_ferro, axis=(1, 2))
    pred_ferro_std = np.std(np.nanmean(pred_ferro, axis=2), axis=1)

    pred_spin = n*thetas_rescaled/(n*thetas_rescaled-1)
    pred_spin[np.where(thetas_rescaled<1./n)] = np.nan
    pred_above_mean = np.nanmean(pred_spin, axis=(1, 2))
    pred_above_std = np.std(np.nanmean(pred_spin, axis=(2)), axis=1)

    # Remove top / bottom 5 samples for each scale
    n_rejects = 1
    cross_blob_without_extremes = deepcopy(cross_blob)
    opt_blob_without_extremes = deepcopy(opt_blob)
    for seed in range(n_seeds):
        for scale_idx in range(n_scales):
            idx_min = np.argsort(cross_blob[seed, scale_idx])[:n_rejects]
            idx_max = np.argsort(cross_blob[seed, scale_idx])[-n_rejects:]
            cross_blob_without_extremes[seed, scale_idx, idx_min] = np.nan
            cross_blob_without_extremes[seed, scale_idx, idx_max] = np.nan

            idx_min = np.argsort(opt_blob[seed, scale_idx])[:n_rejects]
            idx_max = np.argsort(opt_blob[seed, scale_idx])[-n_rejects:]
            opt_blob_without_extremes[seed, scale_idx, idx_min] = np.nan
            opt_blob_without_extremes[seed, scale_idx, idx_max] = np.nan


    opt_mean = np.nanmean(opt_blob_without_extremes, axis=(0, 2))
    opt_std = np.std(np.nanmean(opt_blob_without_extremes, axis=2), axis=(0,))
    cross_mean = np.nanmean(opt_blob_without_extremes, axis=(0, 2))
    cross_std = np.std(np.nanmean(opt_blob_without_extremes, axis=2), axis=(0,))




    id_above_1 = np.where(scales_theory>1.)

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    scales_theory_above_1

    scales_theory_above_1 = scales_theory[id_above_1]
    axes[0].errorbar(scales_theory_above_1, thetas_rescaled_averaged_over_repeats.mean(axis=(1))[id_above_1], yerr=thetas_rescaled_averaged_over_repeats.std(axis=(1))[id_above_1], c='g', label='After normalization')
    axes[0].errorbar(scales_theory_above_1, (max_eigs.mean(axis=(1))[id_above_1] /n)**2, yerr=(max_eigs**2).std(axis=(1))[id_above_1]/(n**2), c='cyan', ls='--', label=r'${c^{tr}_{max}}^2$')
    axes[0].errorbar(scales_theory_above_1, thetas_theory_averaged_over_repeats.mean(axis=(1))[id_above_1], yerr=thetas_theory_averaged_over_repeats.std(axis=(1))[id_above_1], c='k', label='Direct samples')
    axes[0].plot(scales_theory_above_1, (1.-1./scales_theory_above_1)**2, ls='--', c='b', lw=3, label='Theory (ferro)')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'Strength of the couplings $\sigma$')
    axes[0].set_ylabel(r'Overlaps $\theta$')


    axes[1].errorbar(scales_theory, pred_ferro_mean, yerr=pred_ferro_std, c='m', ls='--', label=r'Predicted $\langle\gamma\rangle$ (ferro)')
    axes[1].errorbar(scales_theory, pred_above_mean, yerr=pred_above_std, c='olive', ls='--', label=r'Predicted $\langle\gamma\rangle$ (spin)')
    axes[1].fill_between(scales, cross_mean-cross_std, cross_mean+cross_std, color='g', alpha=.5, label=r'$\gamma^{cross}$ ')
    axes[1].fill_between(scales, opt_mean-opt_std, opt_mean+opt_std, color='r', alpha=.5, label=r'$\gamma^{opt}$')
    axes[1].plot(scales, cross_mean, color='g')
    axes[1].plot(scales, opt_mean, color='r')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].set_xlabel(r'Strength of the couplings $\sigma$')
    axes[1].set_ylabel(r'Regularization $\gamma$')

    plt.savefig(folder + 'full_figure.pdf')
    plt.close()



# need to define it in global scope to be able to call from pool
def run_one_seed(seed, pars):
    scale = pars['scale']
    model_to_fit = CenteredGM(n, seed=seed, precision=pars, silent=True)
    sys.stdout.flush()
    explorer = LikelihoodEstimator(model_to_fit, n_batches=10, name='small_alpha/experiments/' + 'J_{:.2e}/seed{}_N_'.format(scale, seed) + '{}')
    np.save('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/precision.npy'.format(scale, seed, n), model_to_fit.precision)

    n_repeats = 100
    n_samples = 1
    explorer.alpha = n_samples/n

    acc_finite = 0
    acc_infinite = 0

    crosses, opts = np.zeros(n_repeats), np.zeros(n_repeats)
    preds = np.zeros(n_repeats)
    thetas = np.zeros(n_repeats)
    all_thetas = []

    while acc_finite < n_repeats:
        logging.critical('current value of acc_finite : {}'.format(acc_finite))
        gamma_is_infinite = False
        C, s = model_to_fit.get_empirical_C(n_samples, return_samples=True)
        logging.critical('sampled s shape: {}'.format(s.shape))
        u = s / np.sqrt(s.dot(s.T))
        theta = 1/n * u.dot(model_to_fit.covariance.dot(u.T)).item()

        all_thetas.append(theta)

        if theta < 1./n+0.: #+ 1e-7
            acc_infinite += 1
            gamma_is_infinite = True
            logging.critical('theta was too small, expect gamma infinite')
            if scale > 1:
                logging.critical('DANGER: theta was too small for scale {}'.format(scale))
            gamma_pred = scale**(-2)
        else:
            thetas[acc_finite-1] = theta
            acc_finite += 1
            if scale > 1:
                gamma_pred = (theta-1)**2
            else:
                gamma_pred = theta*n / (n*theta -1)

        eigs, BasisChange = tch.symeig(tch.from_numpy(C), eigenvectors=True)
        gammas = np.exp(np.linspace(*np.log([5e-2, 5e1]), 20)) * gamma_pred

        if gamma_is_infinite:
            L_train, L_test, L_gen, logZ, mus = explorer.get_likelihoods(1/n, tch.from_numpy(C), gammas)
        else:
            try:
                out, out_single = explorer.do_one_round(gammas, C_emp=tch.from_numpy(C), find_half_cross=False)
                L_test, L_gen =  np.array([i.item() for i in out['L_test']]), np.array([i.item() for i in out['L_gen']])
                crosses[acc_finite-1] = out_single['gamma_cross']
                opts[acc_finite-1] = out_single['gamma_opt']
                preds[acc_finite-1] = gamma_pred
                logging.critical('Found any step at which L_test was bigger than L_gen? {}'.format(np.any(L_test>L_gen)))
            except Exception as e:
                logging.critical(e)
                raise RuntimeError('Error in likelihoods arose when we expected it to go fine')


        if not gamma_is_infinite or (acc_infinite <2 and gamma_is_infinite):
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            if gamma_is_infinite:
                fig.suptitle('Expect crossing (and optimal) to be infinite')
            else:
                fig.suptitle('Expect crossing (and optimal) to be finite')
            axes[0].plot(gammas, L_test, c='r', label='$L_{train}$ (exp)')
            axes[0].plot(gammas, L_gen, c='g', label='$L_{gen}$ (exp)')
            if scale < 1:
                axes[0].axvline(x=gamma_pred, ls="-", alpha=.2, c='k', label='Predicted crossing ')
            else:
                axes[0].axvline(x=(theta-1)**2, ls="-", alpha=.2, c='k', label='Predicted crossing (ferro)')
            logging.critical('gamma_is_infinite : {}'.format(gamma_is_infinite))
            if not gamma_is_infinite:
                logging.critical('adding the lines')
                axes[0].axvline(x=crosses[acc_finite-1], ls="--", c='m', alpha=.5, label='Empirical crossing')
                axes[0].axvline(x=opts[acc_finite-1], ls="-.", c='orange', alpha=.5, label='Empirical opt')
            axes[0].set_xscale('log')
            axes[0].legend()


            axes[1].plot(gammas, (L_test>L_gen), c='r', label='Test is above')
            if scale < 1:
                axes[1].axvline(x=gamma_pred, ls="--", c='k', label='Predicted crossing ')
            else:
                axes[1].axvline(x=(theta-1)**2, ls="--", c='k', label='Predicted crossing')

            if not gamma_is_infinite:
                axes[0].axvline(x=crosses[acc_finite-1], ls="--", c='m', alpha=.5, label='Empirical crossing')
                axes[0].axvline(x=opts[acc_finite-1], ls="-.", c='orange', alpha=.5, label='Empirical opt')
            axes[1].set_xscale('log')
            axes[1].legend()
            if gamma_is_infinite:
                plt.savefig('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/infinite_{}_likelihoods.pdf'.format(scale, seed, n, acc_infinite))
            else:
                plt.savefig('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/finite_{}_likelihoods.pdf'.format(scale, seed, n, acc_finite))

    np.savetxt('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/gamma_opt.txt'.format(scale, seed, n), opts)
    np.savetxt('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/gamma_cross.txt'.format(scale, seed, n), crosses)
    np.savetxt('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/gamma_predictions.txt'.format(scale, seed, n), preds)
    np.savetxt('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/all_thetas.txt'.format(scale, seed, n), all_thetas)
    np.savetxt('out/small_alpha/experiments/J_{:.2e}/seed{}_N_{}/thetas.txt'.format(scale, seed, n), thetas)



def do_experiment():
    n_seeds = 10
    n_scales = 100

    scales_below_1 = np.exp(np.linspace(*np.log([1e-2, .9]), n_scales//2))
    scales_above_1 = np.exp(np.linspace(*np.log([1.1, 100]), n_scales//2))
    scales = np.hstack([scales_below_1, scales_above_1])

    for scale in tqdm(scales):
        params = {'scale': scale, 'sparsity': 1.}
        pool = ThreadPool(n_seeds)
        _ = pool.starmap(run_one_seed, zip(range(n_seeds), [params for _ in range(n_seeds)]))





if __name__ == '__main__':
    # do_theory()
    # do_experiment()
    # do_plots()
