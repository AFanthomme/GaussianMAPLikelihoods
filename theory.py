from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import curve_fit
import json
import os
from copy import deepcopy
import sys
import logging

class LikelihoodEstimator:
    def __init__(self, gaussian_model, alpha=3., n_batches=1, name='tridiag{}'):
        self.ref_model = gaussian_model
        self.N = self.ref_model.dim
        self.name = name.format(self.N)
        self.alpha = alpha

        self.n_batches = n_batches
        self.C_true = tch.from_numpy(self.ref_model.covariance)

        # model.precision is the full precision matrix (mu - J)^true, split it here
        self.J_true = -tch.from_numpy(self.ref_model.precision - np.diag(np.diag(self.ref_model.precision)))
        self.mu_true = np.mean(np.diag(self.ref_model.precision))
        self.logZ_true = (.5 * (self.mu_true - tch.mean(tch.log(self.mu_true - self.J_true)))).item()

        # Commodity to have it here
        self.labels = ['L_train', 'L_test', 'L_gen', 'L_gen_bis', 'Q2', 'logZ', 'mu', 'errors_on', 'errors_off', 'mu_dot', 'gamma_cros_res', 'L_test_dot', 'delta_L']

        # To store all results instead of only mean and std
        # self.full_acc = []
        # self.full_acc_sv = []
        self._make_output_dir()

    def _make_output_dir(self):
        try:
            with open('out/test_out_exists.txt'.format(self.name)) as f:
                pass
            with open('out/{}/test_out_exists.txt'.format(self.name)) as f:
                pass

        except FileNotFoundError:
            os.makedirs('out/{}'.format(self.name), exist_ok=True)
            with open('out/{}/test_out_exists.txt'.format(self.name), mode='w+') as f:
                f.write('plop')

    def j_map(self, eigs, gamma, mu, alpha=None):
        # if alpha is None:
        #     alpha = self.alpha

        # return (alpha*eigs + gamma*mu - tch.sqrt((alpha*eigs - gamma*mu)**2 + 4 * alpha * gamma)) / (2. * gamma)
        # print(alpha * eigs)
        # print( gamma*mu)
        # print(tch.sqrt((alpha*eigs - gamma*mu)**2 + 4 * alpha * gamma))
        # return (alpha*eigs + gamma*mu + tch.sqrt((alpha*eigs - gamma*mu)**2 + 4 * alpha * gamma)) / (2. * gamma)
        return (alpha*eigs + gamma*mu - tch.sqrt((alpha*eigs - gamma*mu)**2 + 4 * alpha * gamma)) / (2. * gamma) # NB: THIS IS OK

    def solve_mu(self, gamma, eigs, alpha=None):
        print('In solve_mu')
        print('\t eigs: {}'.format(eigs))
        print('\t gamma: {}'.format(gamma))
        print('\t alpha: {}'.format(alpha))



        def surrogate(mu):
            # print(mu)
            return 1. - tch.mean(1. / (mu - self.j_map(eigs, gamma, mu, alpha=alpha))).item()

        plt.figure()
        # range = np.linspace(.5, 3*alpha/(alpha-1), 100)
        range_ = np.exp(np.linspace(*np.log([1e-5, 1e5]), 100))
        # eigs = np.exp(np.linspace(*np.log([1e-5, 1e5]), 100))

        # print( (mu_ - self.j_map(eigs[0], gamma, mu_, alpha=alpha)) )
        tmp = [surrogate(x) for x in range_]
        plt.plot(range_, tmp)
        plt.xscale('log')
        plt.ylim(-10, 10)
        plt.axhline(0., ls='--', c='k')
        plt.savefig('debug.pdf')
        #
        # tmp1 = [1. - 1. / (mu_ - self.j_map(eigs, gamma, mu_, alpha=alpha)) for mu_ in zip(range_, tmp)]
        # plt.plot(range_, tmp1)
        # plt.xscale('log')
        # plt.ylim(-.01, .01)
        # plt.axhline(0.)
        # plt.savefig('debug2.pdf')

        # root = brentq(surrogate, .5, 1e5, maxiter=100)
        root = brentq(surrogate, .01, 100, maxiter=100)
        print('\t surrogate value at 1: {}'.format(surrogate(1)))
        print('\t surrogate value at mu_opt: {}'.format(surrogate(root)))

        return root

    def get_likelihoods(self, alpha, C_emp, gamma_range):

        self.alpha = alpha
        # C_emp = tch.from_numpy(self.ref_model.get_empirical_C(alpha * self.N))
        # print(C_emp)
        C, BasisChange = tch.symeig(C_emp, eigenvectors=True)
        # C, BasisChange = tch.linalg.eigh(C_emp)
        C_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.C_true).mm(BasisChange))
        # print(C_hat)
        L_train, L_test, Q2, L_gen, logZ, mus = [], [], [], [], [], []
        for gamma in gamma_range:
            mu = self.solve_mu(gamma, C, alpha=alpha)
            if mu is None:
                return None

            mus.append(mu)

            j_star = self.j_map(C, gamma, mu)
            mu_minus_j = mu - j_star

            # The easy ones :
            L_train.append(.5 * tch.mean(C * j_star))
            Q2.append(.5 * tch.mean(j_star ** 2))
            logZ.append(.5 * (mu - tch.mean(tch.log(mu_minus_j))))
            L_gen.append(L_train[-1] - gamma / self.alpha * Q2[-1])
            L_test.append(0.5 * tch.mean(C_hat * j_star))

        logZ = np.array(logZ)
        L_train = np.array(L_train)
        L_test = np.array(L_test)
        L_gen = np.array(L_gen)
        return L_train-logZ, L_test-logZ, L_gen-logZ, logZ, mus


    # def get_likelihoods(self, J, C_emp, gamma, alpha):
    #     C, BasisChange = tch.symeig(C_emp, eigenvectors=True)
    #     mu = self.solve_mu(gamma, C, alpha=alpha)
    #     j_star = self.j_map(C, gamma, mu, alpha=alpha)   # These are the eigs, need to rotate back to original basis
    #     mu_minus_j = mu - j_star
    #
    #     C_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.C_true.float()).mm(BasisChange))
    #
    #     info = {
    #     'mu': mu,
    #     'Q2': .5 * tch.mean(j_star ** 2),
    #     'L_train': .5 * tch.mean(C * j_star),
    #     'L_test': .5 * tch.mean(C_hat * j_star),
    #     'logZ': .5 * (mu - tch.mean(tch.log(mu_minus_j))),
    #     'alpha': alpha,
    #     'gamma': gamma,
    #     }

    def get_j_star(self, gamma, C_emp=None, alpha=None):
        if alpha is None:
            alpha=self.alpha

        # if C_emp is None:
        #     C_emp = tch.from_numpy(self.ref_model.get_empirical_C(alpha * self.N))
        # else:
        #     pass

        print('In get_j_star')
        print('\t C_emp: {}'.format(C_emp))
        print('\t gamma: {}'.format(gamma))
        print('\t alpha: {}'.format(alpha))

        C, BasisChange = tch.symeig(C_emp, eigenvectors=True)
        print('\t eigs: {}'.format(C))
        mu = self.solve_mu(gamma, C, alpha=alpha)
        print('\t mu: {}'.format(mu))
        j_star = self.j_map(C, gamma, mu, alpha=alpha)   # These are the eigs, need to rotate back to original basis
        mu_minus_j = mu - j_star

        print('\t j_star: {}'.format(j_star))

        C_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.C_true.float()).mm(BasisChange))

        info = {
        'mu': mu,
        'Q2': .5 * tch.mean(j_star ** 2),
        'L_train': .5 * tch.mean(C * j_star),
        'L_test': .5 * tch.mean(C_hat * j_star),
        'logZ': .5 * (mu - tch.mean(tch.log(mu_minus_j))),
        'alpha': alpha,
        'gamma': gamma,
        }

        # print(.5 * tch.mean(j_star ** 2))
        # print('basis change symmetric?', BasisChange - BasisChange.transpose(0, 1))
        # print('two possible basis changes identical?',  BasisChange.mm(tch.diag(j_star)).mm(BasisChange.transpose(0, 1)) - (BasisChange.transpose(0, 1)).mm(tch.diag(j_star)).mm(BasisChange))
        return info, BasisChange.mm(tch.diag(j_star)).mm(BasisChange.transpose(0, 1)) # this is the good one
        # return info, tch.diag(j_star)
        # return info, (BasisChange.transpose(0, 1)).mm(tch.diag(j_star)).mm(BasisChange)


    def do_one_round(self, gamma_range, find_cross=True, find_half_cross=True, C_emp=None):
        # Draw the covariance matrix1
        if C_emp is None:
            C_emp = tch.from_numpy(self.ref_model.get_empirical_C(self.alpha * self.N))
        else:
            pass

        # C is the vector of eigenvalues of C_emp
        C, BasisChange = tch.symeig(C_emp, eigenvectors=True)
        # C, BasisChange = tch.linalg.eigh(C_emp)

        # This part for sanity check purposes

        # B is orthogonal
        # print(B.mm(B.transpose(0, 1)))

        # B is NOT symmetrical
        # print(tch.max(B - B.transpose(0, 1)))

        # B is normed
        # print(B.shape, tch.sum(B**2, dim=0))

        # When alpha is huge, C_emp-C_true is small in infty norm
        # print(tch.max(self.C_true - C_emp))

        # Therefore, in that case, diagonalizing C_emp should almost diagonalize C_true
        # print(B.transpose(0, 1).mm(self.C_true).mm(B))

        # Prepare all accumulators...
        L_train, L_test, Q2, L_gen, L_gen_bis, logZ, mus = [], [], [], [], [], [], []
        delta_L = []
        errors_on, errors_off = [], []
        mu_dot, gamma_cross_res, L_test_dot = [], [], []

        # Pre-compute some stuff to win time (and readability)
        alpha_C = self.alpha * C
        alpha = self.alpha

        # For L_test
        C_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.C_true).mm(BasisChange))

        # For L_gen_bis, compute J in the inference basis (eigs of C_emp)
        J_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.J_true).mm(BasisChange))
        # J_hat = tch.diag(BasisChange.transpose(0, 1).mm((self.mu_true - self.J_true)).mm(BasisChange))
        # J_hat = tch.diag(BasisChange.transpose(0, 1).mm((self.mu_true - self.J_true)).mm(BasisChange))

        # To compute meaningful means
        support_size = tch.sum(tch.where(self.J_true != 0, tch.ones(self.N, self.N).double(), tch.zeros(self.N, self.N).double())).item()

        for gamma in gamma_range:
            mu = self.solve_mu(gamma, C)
            if mu is None:
                return None

            mus.append(mu)

            j_star = self.j_map(C, gamma, mu)
            mu_minus_j = mu - j_star

            # The easy ones :
            L_train.append(.5 * tch.mean(C * j_star))
            Q2.append(.5 * tch.mean(j_star ** 2))
            logZ.append(.5 * (mu - tch.mean(tch.log(mu_minus_j))))
            L_gen.append(L_train[-1] - gamma / self.alpha * Q2[-1])

            C_gen = 1./mu_minus_j # in the inference basis
            # logging.critical(np.sum(np.diag(C_gen.numpy())))
            L_gen_bis.append(.5 * tch.mean(C_gen * J_hat))
            # tmp = .5 * tch.mean(C_gen * J_hat) # compare with the other way around for basis change

            # L_gen_bis.append(.5 * tch.mean(C_gen * j_star)) # to test if we get exactly L_gen in that case


            # L_test is a bother : we know J_map only in the C_emp diagonalizing basis
            # Therefore, need to rotate C_true in the inference basis (keep only diag, rest is not useful)

            L_test.append(0.5 * tch.mean(C_hat * j_star))

            delta_L.append(L_train[-1] - L_test[-1])

            # Now for the model errors (computed in the true model basis:
            J_star_in_true_basis = BasisChange.mm(tch.diag(j_star)).mm(BasisChange.transpose(0, 1))
            J_star_on_support = tch.where(self.J_true != 0, J_star_in_true_basis, tch.zeros(self.N, self.N).double())
            J_star_off_support = tch.where(self.J_true != 0, tch.zeros(self.N, self.N).double(), J_star_in_true_basis)

            C_star_in_true_basis = BasisChange.mm(tch.diag(1./(mu-j_star))).mm(BasisChange.transpose(0, 1))
            # L_gen_bis.append(.5 * self.N * tch.mean(C_star_in_true_basis * self.J_true))
            # logging.critical(tmp-L_gen_bis[-1])


            errors_on.append(tch.sum((J_star_on_support-self.J_true)**2)/support_size)
            errors_off.append(tch.sum((J_star_off_support)**2)/(self.N ** 2 - support_size))

            # Now, the gamma_cross "residual" -> = 0 at crossing
            gamma_cross_res.append(gamma - self.alpha * tch.mean(j_star * (C-C_hat)) / (2*Q2[-1]))

            # Now, the gory stuff: find mu_dot and L_test_dot
            D = np.sqrt((alpha_C-gamma*mu)**2+4*gamma*alpha)

            A = mu - (2*alpha + gamma * (mu**2) - alpha_C * mu)/ D
            A /= (2.*gamma)

            B = 1. - (gamma*mu - alpha_C) / D
            B /= 2.

            _n = tch.mean((A - j_star / gamma) / (mu_minus_j**2))
            _d = tch.mean((1.-B) / (mu_minus_j**2))

            mu_dot.append(_n/_d)

            j_dot = mu_dot[-1] * B - j_star / gamma + A
            j_dot_term = 0.5 * tch.mean(j_dot*C_hat)
            logZ_dot = .5 * (mu_dot[-1] - tch.mean((mu_dot[-1]-j_dot)/(mu_minus_j)))
            L_test_dot.append(j_dot_term - logZ_dot)


        # Use our bad approxs (very low res in gamma) as seed for brentq (in logspace, seems better)
        # and get our gamma_opt th (which is a priori correct given previous study) with great accuracy
        # looking in a 0.1, 10 log window around exp pred seems reasonable

        ## PROTOTYPE
        def surr_opt(log_gamma):
            gamma = np.exp(log_gamma)
            mu = self.solve_mu(gamma, C)
            j_star = self.j_map(C, gamma, mu)
            mu_minus_j = mu - j_star

            D = np.sqrt((alpha_C-gamma*mu)**2+4*gamma*alpha)
            A = mu - (2*alpha + gamma * (mu**2) - alpha_C * mu) / D
            A /= (2.*gamma)
            B = 1. - (gamma*mu - alpha_C) / D
            B /= 2.

            _n = tch.mean((A - j_star / gamma) / (mu_minus_j**2))
            _d = tch.mean((1.-B) / (mu_minus_j**2))
            mu_dot = _n/_d

            j_dot = mu_dot * B - j_star / gamma + A
            j_dot_term = 0.5 * tch.mean(j_dot*C_hat)
            logZ_dot = .5 * (mu_dot - tch.mean((mu_dot-j_dot)/(mu_minus_j)))

            return j_dot_term - logZ_dot

        initial_guess = gamma_range[np.argmax(np.array(L_test)-np.array(logZ))]
        # print(initial_guess)
        try:
            log_gamma_opt = brentq(surr_opt, np.log(0.1*initial_guess), np.log(10*initial_guess), maxiter=100)
        except:
            # log_gamma_opt = brentq(surr_opt, np.log(0.01*initial_guess), np.log(100*initial_guess), maxiter=100)
            logging.critical('error in brentq for gamma_opt')
            log_gamma_opt = initial_guess
        gamma_opt = np.exp(log_gamma_opt)

        # Get L_opt
        m = self.solve_mu(gamma_opt, C)
        m_j = m - self.j_map(C, gamma_opt, m)
        L_opt = 0.5 * (tch.mean(C_hat * j_star) - (mu - tch.mean(tch.log(m_j))))
        # print('L_opt', L_opt)

        def surr_cross(log_gamma):
            gamma = np.exp(log_gamma)
            mu = self.solve_mu(gamma, C)
            j_star = self.j_map(C, gamma, mu)

            L_train = .5 * tch.mean(C * j_star)
            Q2 = .5 * tch.mean(j_star ** 2)
            L_gen = L_train - gamma / self.alpha * Q2
            L_test = 0.5 * tch.mean(C_hat * j_star)

            return (L_gen-L_test)/(L_train-L_test)

        if find_cross:
            try:
                initial_guess = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))))]
                log_gamma_cross = brentq(surr_cross, np.log(0.1*initial_guess), np.log(1e1*initial_guess), maxiter=100)
            except:
                # print('Using alternative gamma cross interval')
                initial_guess = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))))]
                log_gamma_cross = brentq(surr_cross, np.log(0.01*initial_guess), np.log(1e2*initial_guess), maxiter=100)
        else:
            log_gamma_cross = np.log(gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))))])

        gamma_cross = np.exp(log_gamma_cross)

        if find_half_cross:
            def surr_half_cross(log_gamma):
                return surr_cross(log_gamma) - 0.5

            initial_guess = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))-0.5))]
            try:
                initial_guess = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))))]
                log_gamma_half_cross = brentq(surr_half_cross, np.log(0.1*initial_guess), np.log(1e1*initial_guess), maxiter=100)
            except:
                # print('Using alternative gamma cross interval')
                initial_guess = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))))]
                log_gamma_half_cross = brentq(surr_half_cross, np.log(0.01*initial_guess), np.log(1e2*initial_guess), maxiter=100)
            gamma_half_cross = np.exp(log_gamma_half_cross)
        else:
            gamma_half_cross = gamma_range[np.argmin(np.abs((np.array(L_gen)-np.array(L_test))/(np.array(L_train)-np.array(L_test))-0.5))]

        # Wrap results in a dict for cleaner code
        # Keep residuals cause figure is nice, but the good gammas
        out = {}
        quantities = [L_train, L_test, L_gen, L_gen_bis, Q2, logZ, mus, errors_on,
                      errors_off, mu_dot, gamma_cross_res, L_test_dot, delta_L]

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
            # self.full_acc.append(one_round_result)
            # self.full_acc_sv.append(one_round_result_sv)

            if one_round_result is None:
                continue
            if idx == 0:
                accumulator = {key: tch.zeros(len(gamma_range), self.n_batches) for key in one_round_result}
                accumulator_sv = {key: tch.zeros(self.n_batches) for key in one_round_result_sv}

            for key, value in one_round_result.items():
                accumulator[key][:, idx] = tch.tensor(value)

            for key, value in one_round_result_sv.items():
                accumulator_sv[key][idx] = tch.tensor(value)

            idx += 1

            if idx % verbosity == 0:
                print('Finished pass {} over {} \r'.format(idx, self.n_batches))
                sys.stdout.flush()

        return accumulator, accumulator_sv

    def grid_exploration(self, alpha_range, gamma_range, plot=True, verbosity=2, find_cross=True):
        means_acc = {}
        std_acc = {}
        gamma_range_ref = deepcopy(gamma_range)

#         additional_quantities = ['crossing', 'half_crossing', 'lkl_ratio', 'crossings_pred', 'gamma_opt_pred', 'gamma_opt']

        for key in self.labels + ['gamma_half_cross', 'gamma_cross', 'gamma_opt', 'lkl_ratio', 'L_opt']:
            means_acc[key] = []
            std_acc[key] = []


        for alpha in alpha_range:
            self.alpha = alpha
            gamma_range = gamma_range_ref #  alpha *
            acc, acc_sv = self.gamma_exploration(gamma_range, verbosity=verbosity, find_cross=find_cross)

            for key, value in acc.items():
                value = value.cpu().numpy()
                means_acc[key].append(value.mean(axis=-1))
                std_acc[key].append(value.std(axis=-1))

            means_acc['lkl_ratio'].append(tch.mean((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), dim=-1).cpu().numpy())
            std_acc['lkl_ratio'].append(tch.std((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), dim=-1).cpu().numpy())

            for key, value in acc_sv.items():
                value = value.cpu().numpy()
                means_acc[key].append(value.mean())
                std_acc[key].append(value.std())


            if plot:

                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')


                fig, axes = plt.subplots(4, 2, figsize=(12, 13))

                axes[0,0].set_title(r'$\mu$')
                axes[0,0].errorbar(gamma_range, means_acc['mu'][-1], yerr=std_acc['mu'][-1], c='b')
                axes[0,0].set_xscale('log')

                # The multiplication by gamma is here to switch from d gamma to d loggamma
                axes[0,1].set_title(r'Derivatives wrt $log\gamma$')
                axes[0,1].axhline(y=0, c='k', ls='--')
                axes[0,1].errorbar(gamma_range, gamma_range * means_acc['mu_dot'][-1], yerr=gamma_range * std_acc['mu_dot'][-1], c='g', label=r'$\mu$')
                axes[0,1].errorbar(gamma_range, gamma_range * means_acc['L_test_dot'][-1], yerr=gamma_range * std_acc['L_test_dot'][-1], c='r', label=r'$L_{test}$')
                axes[0,1].axvline(x=means_acc['gamma_opt'][-1])
                axes[0,1].set_xscale('log')
                axes[0,1].legend()

                th_acc_substracted_logZ = {}

                # Old
                for key in ['L_train', 'L_test', 'L_gen', 'L_gen_bis']:
                    th_acc_substracted_logZ[key] = means_acc[key][-1] - means_acc['logZ'][-1]

                # for key in ['L_train', 'L_test', 'L_gen']:
                #     th_acc_substracted_logZ[key] = means_acc[key][-1] - means_acc['logZ'][-1]
                #
                # print(means_acc['L_gen_bis'][-1].shape, means_acc['L_gen_bis'][-1])
                # print(self.logZ_true)
                # th_acc_substracted_logZ['L_gen_bis'] = means_acc['L_gen_bis'][-1] - self.logZ_true # In that case, substract the log Z true?

                axes[1,0].set_title('Likelihoods (without logZ)')
                axes[1,0].errorbar(gamma_range, means_acc['L_train'][-1], yerr=std_acc['L_train'][-1], c='b', ls='--', label=r'$L_{train}$')
                axes[1,0].errorbar(gamma_range, means_acc['L_test'][-1], yerr=std_acc['L_test'][-1], c='r', label=r'$L_{test}$')
                axes[1,0].errorbar(gamma_range, means_acc['L_gen'][-1], yerr=std_acc['L_gen'][-1], c='g', label=r'$L_{gen}$')
                axes[1,0].errorbar(gamma_range, means_acc['L_gen_bis'][-1], yerr=std_acc['L_gen_bis'][-1], c='m', ls='-.', label=r'$L_{true \, gen}$')
                axes[1,0].legend()
                axes[1,0].set_xscale('log')

                axes[1,1].set_title('Likelihoods (with logZ)')

                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_train'], yerr=std_acc['L_train'][-1], c='b', ls='--', label='$L_{train}$')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_test'], yerr=std_acc['L_test'][-1], c='r', label=r'$L_{test}$')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_gen'], yerr=std_acc['L_gen'][-1], c='g', label=r'$L_{gen}$')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_gen_bis'], yerr=std_acc['L_gen_bis'][-1], c='m', ls='-.', label=r'$L_{true \, gen}$')
                axes[1,1].axvline(x=means_acc['gamma_opt'][-1], c='k')
                axes[1,1].axvline(x=means_acc['gamma_cross'][-1], c='k', ls='--')
                axes[1,1].axvline(x=means_acc['gamma_half_cross'][-1], c='k', ls=':')
                axes[1,1].legend()

                y_max, y_min = np.max(th_acc_substracted_logZ['L_train']), np.min(th_acc_substracted_logZ['L_test'])
                spread = y_min - y_max

                axes[1,1].errorbar(x=means_acc['gamma_opt'][-1], y=y_min+0.2*spread, xerr=std_acc['gamma_opt'][-1], fmt='o', c='k')
                axes[1,1].errorbar(x=means_acc['gamma_cross'][-1], y=y_min+0.3*spread, xerr=std_acc['gamma_cross'][-1], fmt='o', c='k')
                axes[1,1].errorbar(x=means_acc['gamma_half_cross'][-1], y=y_min+0.4*spread, xerr=std_acc['gamma_half_cross'][-1], fmt='o', c='k')
                axes[1,1].set_xscale('log')

                axes[2,0].set_title('Likelihoods ratio')
                axes[2,0].errorbar(gamma_range, means_acc['lkl_ratio'][-1], yerr=std_acc['lkl_ratio'][-1], c='b')
                axes[2,0].set_xscale('log')

                axes[2,1].set_title('Q2')
                axes[2,1].errorbar(gamma_range, means_acc['Q2'][-1], yerr=std_acc['Q2'][-1], c='b')
                axes[2,1].set_xscale('log')

                axes[3,0].set_title('On-support errors')
                axes[3,0].errorbar(gamma_range, means_acc['errors_on'][-1], yerr=std_acc['errors_on'][-1], c='b')
                axes[3,0].set_xscale('log')

                axes[3,1].set_title('Off-support errors')
                axes[3,1].errorbar(gamma_range, means_acc['errors_off'][-1], yerr=std_acc['errors_off'][-1], c='r')
                axes[3,1].set_xscale('log')


                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(r'Summary for $\alpha$ = {}'.format(alpha)+'\n ')
                fig.savefig('out/{}/full_figs_{:.2f}.pdf'.format(self.name, self.alpha))


        plt.figure()
        plt.title(r'Notable gammas as a function of $\alpha$')
        plt.xlabel(r'$\alpha$')
        plt.errorbar(alpha_range, means_acc['gamma_cross'], yerr=std_acc['gamma_cross'], label=r'$\gamma^{cross}$')
        plt.errorbar(alpha_range, means_acc['gamma_opt'], yerr=std_acc['gamma_opt'], label=r'$\gamma^{opt}$')
        plt.errorbar(alpha_range, means_acc['gamma_half_cross'], yerr=std_acc['gamma_half_cross'], label=r'$\gamma^{half-cross}$')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('out/{}/crossings.pdf'.format(self.name))
        plt.close()

        np.savez('out/{}/likelihoods_means.npz'.format(self.name), **means_acc)
        np.savez('out/{}/likelihoods_std.npz'.format(self.name), **std_acc)
        np.savetxt('out/{}/alphas.txt'.format(self.name), alpha_range)
