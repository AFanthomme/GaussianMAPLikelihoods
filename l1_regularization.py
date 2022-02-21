from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy.optimize import curve_fit
import json
import os
from copy import deepcopy
import sys

# from sklearn.covariance import graph_lasso as graphical_lasso
from sklearn.covariance import graphical_lasso

class L1_regularizedLikelihoodEstimator:
    def __init__(self, gaussian_model, alpha=3., n_batches=1, name='tridiag{}'):
        self.ref_model = gaussian_model
        self.N = self.ref_model.dim
        self.name = name.format(self.N)
        self.alpha = alpha

        self.n_batches = n_batches
        self.C_true = self.ref_model.covariance

        # model.precision is the full precision matrix (mu - J)^true, split it here
        self.J_true = - (self.ref_model.precision - np.diag(np.diag(self.ref_model.precision)))
        self.mu_true = np.mean(np.diag(self.ref_model.precision))

        self.lr = 1e-3
        self.n_steps = 1000

        # Commodity to have it here
        self.labels = ['L_train', 'L_test', 'L_gen', 'Q2', 'logZ', 'mu', 'errors_on', 'errors_off', 'mu_dot', 'gamma_cros_res', 'L_test_dot', 'delta_L', 'L_true_gen']
        self._make_output_dir()

    def _make_output_dir(self):
        os.makedirs('out_L1/{}'.format(self.name), exist_ok=True)

    def infer(self, C_emp, gamma):
        if self.alpha < 1:
            mode = 'lars'
        else:
            mode = 'cd'

        # mode = 'lars'
        mode = 'cd'

        C_pred, J_pred = graphical_lasso(C_emp, gamma, mode=mode, max_iter=1000)
        return C_pred, J_pred

    def compute_mu(self, J_star):
        eigs, _ = np.linalg.eigh(J_star)

        def surrogate(mu):
            return 1. - np.mean(1. / (mu - eigs)).item()
        # print(gamma)
        root = brentq(surrogate, np.max(eigs)+1e-5, 1e10, maxiter=100)
        return root

    def do_one_round(self, gamma_range, find_cross=True):
        start_from = None
        # Draw the covariance matrix1
        C_emp = self.ref_model.get_empirical_C(self.alpha * self.N)
        J_true = -( self.ref_model.precision - np.diag(np.diag(self.ref_model.precision)))

        # Prepare all accumulators...
        L_train, L_test, Q2, L_gen, logZ, mus = [], [], [], [], [], []
        delta_L = []
        errors_on, errors_off = [], []
        mu_dot, gamma_cross_res, L_test_dot = [], [], []
        L_true_gen = []


        for gamma in tqdm(gamma_range):
            C_star, J_star = self.infer(deepcopy(C_emp), gamma)

            mu = np.mean(np.diag(J_star))

            J_star = J_star - mu * np.diag(np.ones(self.N))
            J_star = - J_star

            old_C_star = deepcopy(C_star)

            C_emp_non_diag = C_emp - np.diag(np.diag(C_emp))
            C_star_non_diag = C_star - np.diag(np.diag(C_star))
            J_star_non_diag = J_star - np.diag(np.diag(J_star))
            C_true_non_diag = self.C_true - np.diag(np.diag(self.C_true))

            L_train.append(.5 * self.N * np.mean(C_emp_non_diag * J_star_non_diag))
            Q2.append(.5 * self.N * np.mean((J_star-np.diag(np.diag(J_star))) ** 2))
            j_eigs, _ = np.linalg.eigh(J_star)
            logZ.append(.5 * (mu - np.mean(np.log(mu-j_eigs))))
            mus.append(mu)
            L_gen.append(.5 * self.N * np.mean(C_star_non_diag * J_star))
            L_test.append(.5 * self.N * np.mean(C_true_non_diag * J_star))
            L_true_gen.append(.5 * self.N * np.mean(C_star_non_diag * J_true))

            delta_L.append(L_train[-1] - L_test[-1])

        # Wrap results in a dict for cleaner code
        # Keep residuals cause figure is nice, but the good gammas
        n_gamma = len(gamma_range)
        errors_on, errors_off, mu_dot, gamma_cross_res, L_test_dot = np.zeros(n_gamma), np.zeros(n_gamma), np.zeros(n_gamma), np.zeros(n_gamma), np.zeros(n_gamma),


        # Here, do what we need to estimate the noticeable gammas; this should be very precise, assuming
        gamma_half_cross, gamma_cross, gamma_opt, L_opt = 0.,  0.,  0.,  0.,

        width = 2.
        inv_width = 1./width
        steps = 20

        gamma_opt_est = gamma_range[np.argmax(np.array(L_test)-np.array(logZ))]
        gamma_opt_range = np.exp(np.linspace(np.log(inv_width*gamma_opt_est), np.log(width*gamma_opt_est), steps))
        gamma_cross_est = gamma_opt_est
        gamma_cross_range = np.exp(np.linspace(np.log(inv_width*gamma_cross_est), np.log(width*gamma_cross_est), steps))

        # Half-cross (look for it before opt otherwise it could give large gamma)
        gamma_half_cross_est = gamma_range[np.argmin(((np.array(L_test)+np.array(L_train)-2*np.array(L_gen))**2)[:np.argmax(np.array(L_test)-np.array(logZ))])]
        gamma_half_cross_range = np.exp(np.linspace(np.log(inv_width*gamma_half_cross_est), np.log(width*gamma_half_cross_est), steps))


        for round in range(2,4):
            print(gamma_cross_est)

            L_test_tmp, L_gen_tmp, logZ_tmp = [], [], []

            for gamma in tqdm(gamma_opt_range):
                C_star, J_star = self.infer(deepcopy(C_emp), gamma)
                mu = np.mean(np.diag(J_star))
                J_star = J_star - mu * np.diag(np.ones(self.N))
                J_star = - J_star
                C_emp_non_diag = C_emp - np.diag(np.diag(C_emp))
                C_star_non_diag = C_star - np.diag(np.diag(C_star))
                J_star_non_diag = J_star - np.diag(np.diag(J_star))
                C_true_non_diag = self.C_true - np.diag(np.diag(self.C_true))
                j_eigs, _ = np.linalg.eigh(J_star)
                logZ_tmp.append(.5 * (mu - np.mean(np.log(mu-j_eigs))))
                L_gen_tmp.append(.5 * self.N * np.mean(C_star_non_diag * J_star))
                L_test_tmp.append(.5 * self.N * np.mean(C_true_non_diag * J_star))

            gamma_opt_est = gamma_opt_range[np.argmax(np.array(L_test_tmp)-np.array(logZ_tmp))]
            gamma_opt_range = np.exp(np.linspace(np.log((inv_width**(1./round))*gamma_opt_est), np.log((width**(1./round))*gamma_opt_est), steps))

            L_test_tmp, L_gen_tmp, logZ_tmp = [], [], []

            for gamma in tqdm(gamma_cross_range):
                C_star, J_star = self.infer(deepcopy(C_emp), gamma)
                mu = np.mean(np.diag(J_star))
                J_star = J_star - mu * np.diag(np.ones(self.N))
                J_star = - J_star
                C_emp_non_diag = C_emp - np.diag(np.diag(C_emp))
                C_star_non_diag = C_star - np.diag(np.diag(C_star))
                J_star_non_diag = J_star - np.diag(np.diag(J_star))
                C_true_non_diag = self.C_true - np.diag(np.diag(self.C_true))
                j_eigs, _ = np.linalg.eigh(J_star)
                logZ_tmp.append(.5 * (mu - np.mean(np.log(mu-j_eigs))))
                L_gen_tmp.append(.5 * self.N * np.mean(C_star_non_diag * J_star))
                L_test_tmp.append(.5 * self.N * np.mean(C_true_non_diag * J_star))


            gamma_cross_est = gamma_cross_range[np.argmin(np.abs(np.array(L_test_tmp)-np.array(L_gen_tmp)))]
            gamma_cross_range = np.exp(np.linspace(np.log((inv_width**(1./round))*gamma_cross_est), np.log((width**(1./round))*gamma_cross_est), steps))
            print(gamma_cross_est)


            L_test_tmp, L_gen_tmp, L_train_tmp = [], [], []

            for gamma in tqdm(gamma_half_cross_range):
                C_star, J_star = self.infer(deepcopy(C_emp), gamma)
                mu = np.mean(np.diag(J_star))
                J_star = J_star - mu * np.diag(np.ones(self.N))
                J_star = - J_star
                C_emp_non_diag = C_emp - np.diag(np.diag(C_emp))
                C_star_non_diag = C_star - np.diag(np.diag(C_star))
                J_star_non_diag = J_star - np.diag(np.diag(J_star))
                C_true_non_diag = self.C_true - np.diag(np.diag(self.C_true))
                j_eigs, _ = np.linalg.eigh(J_star)
                L_train_tmp.append(.5 * self.N * np.mean(C_emp_non_diag * J_star_non_diag))
                L_gen_tmp.append(.5 * self.N * np.mean(C_star_non_diag * J_star))
                L_test_tmp.append(.5 * self.N * np.mean(C_true_non_diag * J_star))

            gamma_half_cross_est = gamma_half_cross_range[np.argmin((np.array(L_test_tmp)+np.array(L_train_tmp)-2*np.array(L_gen_tmp))**2)]
            gamma_half_cross_range = np.exp(np.linspace(np.log(inv_width*gamma_half_cross_est), np.log(width*gamma_half_cross_est), steps))

        # Do with log so we get the errors computed in log
        gamma_opt = np.log(gamma_opt_est)
        gamma_cross = np.log(gamma_cross_est)
        gamma_half_cross = np.log(gamma_half_cross_est)

        out = {}
        quantities = [L_train, L_test, L_gen, Q2, logZ, mus, errors_on,
                      errors_off, mu_dot, gamma_cross_res, L_test_dot, delta_L, L_true_gen]

        for key, value in zip(self.labels, quantities):
            out[key] = value

        out_single_values = {}
        single_values_labels = ['gamma_half_cross', 'gamma_cross', 'gamma_opt', 'L_opt']
        for k, v in zip(single_values_labels, [gamma_half_cross, gamma_cross, gamma_opt, L_opt]):
            out_single_values[k] = v

        return out, out_single_values

    def gamma_exploration(self, gamma_range, verbosity=2, find_cross=True):
        idx = 0
        while idx < self.n_batches:
            one_round_result, one_round_result_sv = self.do_one_round(gamma_range, find_cross=find_cross)
            if one_round_result is None:
                continue
            if idx == 0:
                accumulator = {key: np.zeros((len(gamma_range), self.n_batches)) for key in one_round_result}
                accumulator_sv = {key: np.zeros(self.n_batches) for key in one_round_result_sv}

            for key, value in one_round_result.items():
                accumulator[key][:, idx] = np.array(value)

            for key, value in one_round_result_sv.items():
                accumulator_sv[key][idx] = np.array(value)

            idx += 1

            if idx % verbosity == 0:
                print('Finished pass {} over {} \r'.format(idx, self.n_batches))
                sys.stdout.flush()

        return accumulator, accumulator_sv

    def grid_exploration(self, alpha_range, gamma_range, plot=True, verbosity=2, find_cross=True):
        means_acc = {}
        std_acc = {}
        gamma_range_ref = deepcopy(gamma_range)

        for key in self.labels + ['gamma_half_cross', 'gamma_cross', 'gamma_opt', 'lkl_ratio', 'L_opt']:
            means_acc[key] = []
            std_acc[key] = []


        for alpha in alpha_range:
            self.alpha = alpha
            gamma_range = gamma_range_ref 
            acc, acc_sv = self.gamma_exploration(gamma_range, verbosity=verbosity, find_cross=find_cross)

            for key, value in acc.items():
                means_acc[key].append(value.mean(axis=-1))
                std_acc[key].append(value.std(axis=-1))

            means_acc['lkl_ratio'].append(np.mean((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), axis=-1))
            std_acc['lkl_ratio'].append(np.std((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), axis=-1))

            for key, value in acc_sv.items():
                means_acc[key].append(value.mean())
                std_acc[key].append(value.std())

            np.savetxt('out_L1/{}/alpha_{:.2f}_gamma_estimations_mean.txt'.format(self.name, self.alpha), np.vstack([means_acc['gamma_cross'], means_acc['gamma_opt']]))
            np.savetxt('out_L1/{}/alpha_{:.2f}_gamma_estimations_std.txt'.format(self.name, self.alpha), np.vstack([std_acc['gamma_cross'], std_acc['gamma_opt']]))

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            if plot:
                fig, axes = plt.subplots(1, 1, figsize=(6, 6))

                axes.set_title('Likelihoods (with logZ)')
                axes.errorbar(gamma_range, means_acc['L_train'][-1] - means_acc['logZ'][-1], yerr=std_acc['L_train'][-1], c='b', label=r'$L_{train}$')
                axes.errorbar(gamma_range, means_acc['L_test'][-1] - means_acc['logZ'][-1], yerr=std_acc['L_test'][-1], c='r', label=r'$L_{test}$')
                axes.errorbar(gamma_range, means_acc['L_gen'][-1] - means_acc['logZ'][-1], yerr=std_acc['L_gen'][-1], c='g', label=r'$L_{gen}$')
                axes.axvspan(np.exp(np.array(means_acc['gamma_cross'])[-1]-np.array(std_acc['gamma_cross']))[-1], np.exp(np.array(means_acc['gamma_cross'])+np.array(std_acc['gamma_cross']))[-1], color='orange', alpha=.5)
                axes.axvline(np.exp(np.array(means_acc['gamma_cross']))[-1], color='orange', alpha=.5, ls='--', label=r'$\gamma^{cross}$')
                axes.axvspan(np.exp(np.array(means_acc['gamma_opt'])-np.array(std_acc['gamma_opt']))[-1], np.exp(np.array(means_acc['gamma_opt'])+np.array(std_acc['gamma_opt']))[-1], color='fuchsia', alpha=.5)
                axes.axvline(np.exp(np.array(means_acc['gamma_opt']))[-1], color='fuchsia', alpha=.5, ls='-.', label=r'$\gamma^{opt}$')
                axes.axvspan(np.exp(np.array(means_acc['gamma_half_cross'])[-1]-np.array(std_acc['gamma_half_cross']))[-1], np.exp(np.array(means_acc['gamma_half_cross'])+np.array(std_acc['gamma_half_cross']))[-1], color='turquoise', alpha=.5)
                axes.axvline(np.exp(np.array(means_acc['gamma_half_cross']))[-1], color='turquoise', alpha=.5, ls='--', label=r'$\gamma^{half-cross}$')
                axes.legend()
                axes.set_xscale('log')
                fig.suptitle(r'Likelihoods for $\alpha$ = {}'.format(alpha)+'\n ')
                fig.savefig('out_L1/{}/likelihoods_{:.2f}.pdf'.format(self.name, self.alpha))


                fig, axes = plt.subplots(1, 1, figsize=(6, 6))
                axes.errorbar(gamma_range, means_acc['L_test'][-1], yerr=std_acc['L_test'][-1], c='r', label=r'$L_{test}$')
                axes.errorbar(gamma_range, means_acc['L_true_gen'][-1], yerr=std_acc['L_true_gen'][-1], c='g', label=r'$L_{true\, gen}$')
                axes.axvspan(np.exp(np.array(means_acc['gamma_half_cross'])[-1]-np.array(std_acc['gamma_half_cross']))[-1], np.exp(np.array(means_acc['gamma_half_cross'])+np.array(std_acc['gamma_half_cross']))[-1], color='orange', alpha=.5)
                axes.axvline(np.exp(np.array(means_acc['gamma_half_cross']))[-1], color='orange', alpha=.5, ls='--', label=r'$\gamma^{half-cross}$')
                axes.legend()
                axes.set_xscale('log')
                fig.savefig('out_L1/{}/gamma_half_cross_study_{:.2f}.pdf'.format(self.name, self.alpha))

                fig, axes = plt.subplots(1, 1, figsize=(10, 10))
                axes.set_title('Mus')
                axes.errorbar(gamma_range, means_acc['mu'][-1], yerr=std_acc['mu'][-1], c='b')
                axes.set_xscale('log')
                fig.suptitle(r'Mu evolution $\alpha$ = {}'.format(alpha)+'\n ')
                fig.savefig('out_L1/{}/mus_{:.2f}.pdf'.format(self.name, self.alpha))

                fig, axes = plt.subplots(1, 1, figsize=(6, 6))
                axes.set_title('Q2')
                axes.errorbar(gamma_range, means_acc['Q2'][-1], yerr=std_acc['Q2'][-1], c='b')
                axes.set_xscale('log')
                fig.suptitle(r'Q2 evolution $\alpha$ = {}'.format(alpha)+'\n ')
                fig.savefig('out_L1/{}/Q2_{:.2f}.pdf'.format(self.name, self.alpha))
