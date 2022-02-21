# Take the results from all threads in a single experiment to make a better estimation (since our method is a bit dirty)
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


def aggregate_optimal_gammas_sparsity_one():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    name='sparsity_one'
    leg_scale_name = r'$\sigma$={:.1e}'
    n_seeds = 4
    N = 200
    scales = [.1, .3, .7, .8, .9]
    j2 = np.array(scales) ** 2 / N

    alphas = np.exp(np.linspace(np.log(3.), np.log(100), 30))

    gamma_crosss = np.zeros((n_seeds, len(scales), len(alphas)))
    gamma_opts = np.zeros((n_seeds, len(scales), len(alphas)))


    colors = cm.viridis(np.linspace(0, 1, len(scales)))
    colors_desaturated = [desaturate(c, .7) for c in colors]

    for seed in range(n_seeds):
        for scale_idx, scale in enumerate(scales):
            # for alpha_idx, alpha in enumerate(alphas):
                gammas = np.loadtxt('out_L1/sparsity_one/J_{:.2e}/seed{}_N_{}/alpha_{:.2f}_gamma_estimations_mean.txt'.format(scale, seed, N, alphas[-1]))
                gamma_crosss[seed, scale_idx, :] = np.exp(gammas[0, :])
                gamma_opts[seed, scale_idx, :] = np.exp(gammas[1, :])


    gamma_opt_mean = np.mean(gamma_opts, axis=0)
    gamma_opt_std = np.std(gamma_opts, axis=0)

    gamma_cross_mean = np.mean(gamma_crosss, axis=0)
    gamma_cross_std = np.std(gamma_crosss, axis=0)

    fig, ax = plt.subplots(1)
    legend_elements = [Line2D([0], [0], color='k', ls='', markerfacecolor='k', marker='x', label=r'$\gamma^{opt}$'),
               Line2D([0], [0], marker='+', color='k', ls='', label=r'$\gamma^{cr}$', markerfacecolor='k', markersize=10),
               # Line2D([0], [0], color='k', label=r'$ 1 /(N \langle J^2 \rangle)$', markerfacecolor='k', markersize=10)
               ]
    for idx, scale in enumerate(scales):
        ax.plot(alphas, alphas*gamma_cross_mean[idx], c=colors[idx], ls='', marker='x')
        ax.plot(alphas, alphas*gamma_opt_mean[idx], c=colors[idx], ls='', marker='+')
        legend_elements.append(Line2D([0], [0], marker= 'o', ls='', color=colors[idx], label=leg_scale_name.format(scale)))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Sampling ratio $\alpha$')
    ax.set_ylabel(r'Regularization $\gamma$')
    ax.set_title(r'Noticeable regularizations $\gamma^{opt}$ and $\gamma^{cross}$')
    fig.savefig('out_L1/sparsity_one/gammas_summary.pdf')

if __name__ == '__main__':
    aggregate_optimal_gammas_sparsity_one()
