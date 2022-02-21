import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from gaussian_model import CenteredGM
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from seaborn import desaturate

def post_run(code='sparsity_one'):
    N = 1000
    n_seeds=8


    alphas = np.concatenate((np.array([0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 25)))) 
    gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 50))

    if code == 'sparsity_one':
        short_base = 'out/sparsity_one/'

        base = short_base + 'J_{:.2e}/seed{}_N_{}/'
        leg_scale_name = r'$\sigma$={:.1e}'
        scales = [.1, .3, .5, .7, 0.8, 0.9]
        j2 = np.array(scales) ** 2 / N # For gamma opt pred

    elif code == 'tridiag':
        short_base = 'out/tridiag/'
        base = short_base + 'J_{:.2e}/seed{}_N_{}/'
        leg_scale_name = r'$j$={:.1e}'
        scales = [.1, .3, .5, .7, 1., 3., 10., 100.]

        j2 = 2* np.array(scales) ** 2 / N # For gamma opt pred

    mu_true_acc = np.zeros((len(scales), len(alphas), n_seeds))
    mu_acc = np.zeros((len(scales), len(alphas), n_seeds))
    gamma_cross_acc = np.zeros((len(scales), len(alphas), n_seeds))
    gamma_opt_acc = np.zeros((len(scales), len(alphas), n_seeds))
    gamma_half_acc = np.zeros((len(scales), len(alphas), n_seeds))
    L_opt_acc = np.zeros((len(scales), len(alphas), n_seeds))
    max_L_train_acc = np.zeros((len(scales), len(alphas), n_seeds))
    deltaL_opt_acc = np.zeros((len(scales), len(alphas), n_seeds))


    deltaL_acc = np.zeros((len(scales), len(alphas), len(gammas), n_seeds))
    L_test_acc = np.zeros((len(scales), len(alphas), len(gammas), n_seeds))

    for scale_idx, scale in enumerate(scales):
        for seed in range(n_seeds):
            folder_name = base.format(scale, seed, N)
            precision = np.load(folder_name + 'precision.npy')
            mean_blob = np.load(folder_name + 'likelihoods_means.npz')
            mu_true = precision[0,0]
            assert np.all(np.diag(precision) == mu_true)
            mu_true_acc[scale_idx, :, seed] += mu_true
            mu_acc[scale_idx, :, seed] = mean_blob['mu'][:, 0]
            gamma_cross_acc[scale_idx, :, seed] = mean_blob['gamma_cross']
            gamma_opt_acc[scale_idx, :, seed] = mean_blob['gamma_opt']
            gamma_half_acc[scale_idx, :, seed] = mean_blob['gamma_half_cross']
            L_opt_acc[scale_idx, :, seed] += np.max(mean_blob['L_test']-mean_blob['logZ'], axis=1)
            max_L_train_acc[scale_idx, :, seed] += np.max(mean_blob['L_train']-mean_blob['logZ'], axis=1)
            plop = np.argmax(mean_blob['L_test']-mean_blob['logZ'], axis=1)
            for alpha_idx in range(len(alphas)):
                deltaL_opt_acc[scale_idx, alpha_idx, seed] += mean_blob['delta_L'][alpha_idx, plop[alpha_idx]]

            deltaL_acc[scale_idx, :, :, seed] += np.max(mean_blob['L_train']-mean_blob['logZ'], axis=1).reshape((len(alphas), 1))-mean_blob['L_test']+mean_blob['logZ']
            L_test_acc[scale_idx, :, :, seed] += mean_blob['L_test']-mean_blob['logZ']

    mu_mean = np.mean(mu_acc, axis=2)
    mu_std = np.std(mu_acc, axis=2)
    one_over_one_minus_mu_std = np.std(1./(1.-mu_acc), axis=2)

    mu_true_mean = np.mean(mu_true_acc, axis=2)
    mu_true_std = np.std(mu_true_acc, axis=2)
    one_over_one_minus_mu_true_std = np.std(1./(1.-mu_true_acc), axis=2)

    print(gamma_cross_acc[0, 0])
    gamma_cross_mean = np.mean(gamma_cross_acc, axis=2)
    gamma_cross_std = np.std(gamma_cross_acc, axis=2)
    print(gamma_cross_mean[0, 0])
    print(gamma_cross_std[0, 0])

    gamma_opt_mean = np.mean(gamma_opt_acc, axis=2)
    gamma_opt_std = np.std(gamma_opt_acc, axis=2)

    gamma_half_mean = np.mean(gamma_half_acc, axis=2)
    gamma_half_std = np.std(gamma_half_acc, axis=2)

    L_opt_mean = np.mean(L_opt_acc, axis=2)
    L_opt_std = np.std(L_opt_acc, axis=2)

    max_L_train_mean = np.mean(max_L_train_acc, axis=2)
    max_L_train_std = np.std(max_L_train_acc, axis=2)

    deltaL_opt_mean = np.mean(deltaL_opt_acc, axis=2)
    deltaL_opt_std = np.std(deltaL_opt_acc, axis=2)

    deltaL_mean = np.mean(deltaL_acc, axis=3)
    deltaL_std = np.std(deltaL_acc, axis=3)

    L_test_mean = np.mean(L_test_acc, axis=3)
    L_test_std = np.std(L_test_acc, axis=3)

    colors = cm.viridis(np.linspace(0, 1, len(scales)))

    print(colors)
    colors_desaturated = [desaturate(c, .7) for c in colors]
    print(colors_desaturated)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')



    fig, ax = plt.subplots(1)
    for idx, scale in enumerate(scales):
        ax.plot(alphas, L_opt_mean[idx]-max_L_train_mean[idx], c=colors[idx], label=leg_scale_name.format(scale))
        ax.fill_between(alphas, L_opt_mean[idx]-max_L_train_mean[idx] -L_opt_std[idx], L_opt_mean[idx]-max_L_train_mean[idx] +L_opt_std[idx], color=colors_desaturated[idx], alpha=.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xscale('log')
    ax.set_ylabel(r'Likelihood gap')
    ax.set_xlabel(r'Sampling ratio $\alpha$')
    ax.set_title(r'Likelihood gap $\Delta L$')
    fig.savefig(short_base + '/L_opt_minus_L_tr.pdf')

    fig, ax = plt.subplots(1)
    legend_elements = [Line2D([0], [0], color='k', ls='', markerfacecolor='k', marker='x', label=r'$\gamma^{opt}$'),
               Line2D([0], [0], marker='+', color='k', ls='', label=r'$\gamma^{cr}$', markerfacecolor='k', markersize=10),
               Line2D([0], [0], color='k', label=r'$ 1 /(N \langle J^2 \rangle)$', markerfacecolor='k', markersize=10)
               ]
    for idx, scale in enumerate(scales):
        ax.errorbar(alphas, gamma_cross_mean[idx], yerr=gamma_cross_std[idx], c=colors[idx], ls='', fmt='x')
        ax.errorbar(alphas, gamma_opt_mean[idx], yerr=gamma_opt_std[idx], c=colors[idx], ls='', fmt='+')
        ax.axhline(y=1./(N*j2[idx]), c=colors[idx], ls='--')
        legend_elements.append(Line2D([0], [0], marker= 'o', ls='', color=colors[idx], label=leg_scale_name.format(scale)))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sampling ratio $\alpha$')
    ax.set_ylabel(r'Regularization $\gamma$')
    ax.set_title(r'Noticeable regularizations $\gamma^{opt}$ and $\gamma^{cross}$')
    fig.savefig(short_base + '/gamma_opt_prediction.pdf')























def post_run_bands():
    N = 1000
    n_seeds=8
    #
    # alphas = np.concatenate((np.array([0.001, 0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 50)))) #25
    # gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 60))

    alphas = np.concatenate((np.array([0.005, 0.01, 0.03, 0.06]), np.exp(np.linspace(*np.log([0.1, 100]), 25)))) #25
    gammas = np.exp(np.linspace(*np.log([5e-4, 1e4]), 50))

    scales = [.1, .3, .7]
    widths = [1, 5, 10, 20, 40, 100, 300, 500]

    short_base = 'out/bands/'
    # short_base = 'out_copy_for_tests/sparsity_one/'

    base = short_base + 'J_{:.2e}_width_{}/seed{}_N_{}/'
    # leg_scale_name = r'$\sigma$={:.1e}'
    j2 = np.array(scales) ** 2 / N # For gamma opt pred



    # mu_true_acc = np.zeros((len(scales), len(widths), len(alphas), n_seeds))
    # mu_acc = np.zeros((len(scales), len(alphas), n_seeds))
    gamma_cross_acc = np.zeros((len(scales), len(widths), len(alphas), n_seeds))
    gamma_opt_acc = np.zeros((len(scales), len(widths), len(alphas), n_seeds))
    gamma_half_acc = np.zeros((len(scales), len(widths), len(alphas), n_seeds))
    # L_opt_acc = np.zeros((len(scales), len(alphas), n_seeds))
    # max_L_train_acc = np.zeros((len(scales), len(alphas), n_seeds))
    # deltaL_opt_acc = np.zeros((len(scales), len(alphas), n_seeds))

    # For 4d plot, taken wrt zero reg
    # deltaL_acc = np.zeros((len(scales), len(alphas), len(gammas), n_seeds))
    # L_test_acc = np.zeros((len(scales), len(alphas), len(gammas), n_seeds))

    for scale_idx, scale in enumerate(scales):
        for width_idx, width in enumerate(widths):
            for seed in range(n_seeds):
                folder_name = base.format(scale, width, seed, N)
                precision = np.load(folder_name + 'precision.npy')
                mean_blob = np.load(folder_name + 'likelihoods_means.npz')
                # mu_true = precision[0,0]
                # assert np.all(np.diag(precision) == mu_true)
                # mu_true_acc[scale_idx, :, seed] += mu_true
                # mu_acc[scale_idx, :, seed] = mean_blob['mu'][:, 0]
                gamma_cross_acc[scale_idx, width_idx, :, seed] = mean_blob['gamma_cross']
                gamma_opt_acc[scale_idx, width_idx, :, seed] = mean_blob['gamma_opt']
                gamma_half_acc[scale_idx, width_idx, :, seed] = mean_blob['gamma_half_cross']
                # L_opt_acc[scale_idx, :, seed] += np.max(mean_blob['L_test']-mean_blob['logZ'], axis=1)
                # max_L_train_acc[scale_idx, :, seed] += np.max(mean_blob['L_train']-mean_blob['logZ'], axis=1)
                # plop = np.argmax(mean_blob['L_test']-mean_blob['logZ'], axis=1)
                # for alpha_idx in range(len(alphas)):
                #     deltaL_opt_acc[scale_idx, alpha_idx, seed] += mean_blob['delta_L'][alpha_idx, plop[alpha_idx]]
                #
                # deltaL_acc[scale_idx, :, :, seed] += np.max(mean_blob['L_train']-mean_blob['logZ'], axis=1).reshape((len(alphas), 1))-mean_blob['L_test']+mean_blob['logZ']
                # L_test_acc[scale_idx, :, :, seed] += mean_blob['L_test']-mean_blob['logZ']

    # mu_mean = np.mean(mu_acc, axis=2)
    # mu_std = np.std(mu_acc, axis=2)
    # one_over_one_minus_mu_std = np.std(1./(1.-mu_acc), axis=2)
    #
    # mu_true_mean = np.mean(mu_true_acc, axis=2)
    # mu_true_std = np.std(mu_true_acc, axis=2)
    # one_over_one_minus_mu_true_std = np.std(1./(1.-mu_true_acc), axis=2)

    print(gamma_cross_acc[0, 0])
    gamma_cross_mean = np.mean(gamma_cross_acc, axis=3)
    gamma_cross_std = np.std(gamma_cross_acc, axis=3)
    # print(gamma_cross_mean[0, 0])
    # print(gamma_cross_std[0, 0])

    gamma_opt_mean = np.mean(gamma_opt_acc, axis=3)
    gamma_opt_std = np.std(gamma_opt_acc, axis=3)

    gamma_half_mean = np.mean(gamma_half_acc, axis=3)
    gamma_half_std = np.std(gamma_half_acc, axis=3)

    # L_opt_mean = np.mean(L_opt_acc, axis=2)
    # L_opt_std = np.std(L_opt_acc, axis=2)
    #
    # max_L_train_mean = np.mean(max_L_train_acc, axis=2)
    # max_L_train_std = np.std(max_L_train_acc, axis=2)
    #
    # deltaL_opt_mean = np.mean(deltaL_opt_acc, axis=2)
    # deltaL_opt_std = np.std(deltaL_opt_acc, axis=2)
    #
    # deltaL_mean = np.mean(deltaL_acc, axis=3)
    # deltaL_std = np.std(deltaL_acc, axis=3)
    #
    # L_test_mean = np.mean(L_test_acc, axis=3)
    # L_test_std = np.std(L_test_acc, axis=3)

    colors = cm.viridis(np.linspace(0, 1, len(widths)))
    symbols = ['']

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(1)
    legend_elements = [Line2D([0], [0], color='k', ls='', markerfacecolor='k', marker='x', label=r'$\gamma^{opt}$'),
               Line2D([0], [0], marker='+', color='k', ls='', label=r'$\gamma^{cr}$', markerfacecolor='k', markersize=10),
               Line2D([0], [0], color='k', label=r'$ 1 /(N \langle J^2 \rangle)$', markerfacecolor='k', markersize=10)
               ]

    for idx, scale in enumerate(scales):
        for width_idx, width in enumerate(widths):
            ax.plot(alphas, gamma_cross_mean[idx, width_idx], c=colors[width_idx], ls='', marker='x')
            ax.plot(alphas, gamma_opt_mean[idx, width_idx], c=colors[width_idx], ls='', marker='+')
            if width_idx == 0:
                ax.axhline(y=1./(N*j2[idx]), c='k', ls='--')
            if idx == 0:
                legend_elements.append(Line2D([0], [0], marker= 'o', ls='', color=colors[width_idx], label=r'$w={}$'.format(width)))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sampling ratio $\alpha$')
    ax.set_ylabel(r'Regularization $\gamma$')
    # ax.set_title(r'Noticeable regularizations $\gamma^{opt}$ and $\gamma^{cross}$ as a function of $\alpha$')
    ax.set_title(r'Noticeable regularizations $\gamma^{opt}$ and $\gamma^{cross}$')
    fig.savefig(short_base + '/gamma_opt_prediction.pdf')




if __name__ == '__main__':
    # post_run('sparsity_one')
    # post_run('tridiag')
    post_run_bands()
