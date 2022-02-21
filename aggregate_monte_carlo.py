import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy

PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"
PASTEL_ORANGE = "#f5b041"



def plot_mean_std(ax, data, axis=0, c_line='g', c_fill=PASTEL_GREEN, label=None, log_yscale=False, rasterized=False):
    if not log_yscale:
        mean =  data.mean(axis=axis)
        std = data.std(axis=axis)
        low = mean - std
        high = mean + std
    else:
        ax.set_yscale('log')
        log_mean = np.nanmean(np.log(data), axis=axis)
        log_std = np.nanstd(np.log(data), axis=axis)
        mean = np.exp(log_mean)
        low = np.exp(log_mean-log_std)
        high = np.exp(log_mean+log_std)

    x = range(mean.shape[0])

    if label is None:
        ax.plot(x, mean, c=c_line, rasterized=rasterized)
    else:
        ax.plot(x, mean, c=c_line, label=label, rasterized=rasterized)

    ax.fill_between(x, low, high, color=c_fill, alpha=.7, zorder=1, rasterized=rasterized)

def make_aggregates(folder, n_seeds=8, t_max=500000, test_every=100):

    train_energy_blob = np.zeros((n_seeds, t_max//test_every+1))
    test_energy_blob = np.zeros((n_seeds, t_max // test_every + 1))
    displacement_blob = np.zeros((n_seeds, t_max // test_every + 1))
    distance_blob = np.zeros((n_seeds, t_max // test_every + 1))

    E_star_test = np.zeros((n_seeds))
    E_star_train = np.zeros((n_seeds))
    E_init_test = np.zeros((n_seeds))
    E_init_train = np.zeros((n_seeds))

    for seed in range(n_seeds):
        E_star_train[seed] = np.loadtxt('{}/seed_{}/E_star_train.txt'.format(folder, seed))
        E_star_test[seed] = np.loadtxt('{}/seed_{}/E_star_test.txt'.format(folder, seed))
        E_init_train[seed] = np.loadtxt('{}/seed_{}/E_train_init.txt'.format(folder, seed))
        E_init_test[seed] = np.loadtxt('{}/seed_{}/E_test_init.txt'.format(folder, seed))
        train_energy_blob[seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))  - E_star_train[seed]
        test_energy_blob[seed] = np.load('{}/seed_{}/test_energy_acc.npy'.format(folder, seed)) - E_star_test[seed]
        distance_blob[seed] = np.load('{}/seed_{}/distance_to_map_acc.npy'.format(folder, seed))
        displacement_blob[seed] = np.load('{}/seed_{}/distance_to_init_acc.npy'.format(folder, seed))


    train_mean = np.mean(train_energy_blob, axis=0)[1:]
    train_std = np.std(train_energy_blob, axis=0)[1:]

    train_min = np.min(train_energy_blob, axis=0)[1:]
    train_max = np.max(train_energy_blob, axis=0)[1:]

    abs_train_min = np.min(np.abs(train_energy_blob), axis=0)[1:]
    abs_train_max = np.max(np.abs(train_energy_blob), axis=0)[1:]

    test_min = np.min(test_energy_blob, axis=0)[1:]
    test_max = np.max(test_energy_blob, axis=0)[1:]

    abs_test_min = np.min(np.abs(test_energy_blob), axis=0)[1:]
    abs_test_max = np.max(np.abs(test_energy_blob), axis=0)[1:]

    test_mean = np.mean(test_energy_blob, axis=0)[1:]
    test_std = np.std(test_energy_blob, axis=0)[1:]

    distance_mean = np.mean(distance_blob, axis=0)[1:]
    distance_std = np.std(distance_blob, axis=0)[1:]

    displacement_mean = np.mean(displacement_blob, axis=0)[1:]
    displacement_std = np.std(displacement_blob, axis=0)[1:]


    np.save('{}/E_train.npy'.format(folder), train_energy_blob)
    np.save('{}/E_train_avg.npy'.format(folder), train_mean)
    np.save('{}/E_train_std.npy'.format(folder), train_std)

    np.save('{}/E_test.npy'.format(folder), test_energy_blob)
    np.save('{}/E_test_avg.npy'.format(folder), test_mean)
    np.save('{}/E_test_std.npy'.format(folder), test_std)

    np.save('{}/distance.npy'.format(folder), distance_blob)
    np.save('{}/distance_avg.npy'.format(folder), distance_mean)
    np.save('{}/distance_std.npy'.format(folder), distance_std)

    fig, ax = plt.subplots()
    plt.title(folder.split('/')[-1])

    plot_mean_std(ax, test_energy_blob[:,1:], axis=0, c_line='b', c_fill=PASTEL_BLUE, label='E-E_star (test)', log_yscale=False)
    plt.legend()
    plt.savefig('{}/energy_test.pdf'.format(folder))
    plt.close()

    fig, axes = plt.subplots(1,2, figsize=(11,5))
    plt.title(folder.split('/')[-1])
    ax = axes[0]
    plot_mean_std(ax, train_energy_blob[:,1:], axis=0, c_line='g', c_fill=PASTEL_GREEN, label='E-E_star (train) (positive values)', log_yscale=True)
    ax.set_yscale('log')
    ax = axes[1]
    tmp = deepcopy(train_energy_blob)
    tmp[np.where(tmp > 0)] = 0.
    tmp = -tmp
    plot_mean_std(ax, tmp[:,1:], axis=0, c_line='r', c_fill=PASTEL_RED, label='E-E_star (train) (negative values)', log_yscale=True)

    ax.set_yscale('log')
    plt.legend()
    plt.savefig('{}/energy_train.pdf'.format(folder))
    plt.close()

    fig, axes = plt.subplots(1,2, figsize=(11,5))
    ax = axes[0]
    plot_mean_std(ax, distance_blob[:,1:], axis=0, c_line='g', c_fill=PASTEL_GREEN, label='Distance to MAP', log_yscale=True)
    plot_mean_std(ax, displacement_blob[:,1:], axis=0, c_line='r', c_fill=PASTEL_RED, label='Distance to init', log_yscale=True)
    ax.legend()
    ax.set_yscale('log')

    fig.savefig('{}/distances.pdf'.format(folder))
    plt.close()







def plot_J_n_2(base_folder, n_seeds=8, t_max=500000, test_every=100):

    for folder in [base_folder+'seed_{}/'.format(seed) for seed in range(n_seeds)]:
        J_acc = np.load(folder+'J_acc.npy')
        print(J_acc.shape)
        J_star = np.loadtxt(folder+'J_star.txt')
        J_init = np.loadtxt(folder+'J_init.txt')

        plt.figure()
        fig, axes = plt.subplots(1,2,figsize=(12, 5))
        ax = axes[0]
        ax.plot(range(t_max//test_every +1), J_acc[:, 0], label='J_00', c='r')
        ax.axhline(J_star[0,0], c='r', ls='--')
        ax.plot(range(t_max//test_every +1), J_acc[:, 1], label='J_01', c='g')
        ax.axhline(J_star[0,1], c='g', ls='--')
        ax.plot(range(t_max//test_every +1), J_acc[:, 2], label='J_11', c='b')
        ax.axhline(J_star[1,1], c='b', ls='--')
        ax.plot(range(t_max//test_every +1), J_acc[:, 3], label='J_10', c='m')
        ax.axhline(J_star[1,0], c='m', ls='--')
        ax.legend()
        ax = axes[1]
        dist = (J_acc[:, 0]-J_star[0,0])**2 + (J_acc[:, 1]-J_star[0,1])**2 + (J_acc[:, 2]-J_star[1,1])**2 + (J_acc[:, 3]-J_star[1,0])**2
        dist = np.sqrt(dist/4)

        dist_init = (J_acc[:, 0]-J_init[0,0])**2 + (J_acc[:, 1]-J_init[0,1])**2 + (J_acc[:, 2]-J_init[1,1])**2 + (J_acc[:, 3]-J_init[1,0])**2
        dist_init = np.sqrt(dist_init/4)
        ax.plot(range(t_max//test_every +1), dist, label='Distance to map', c='orange')
        ax.plot(range(t_max//test_every +1), dist_init, label='Distance to init', c='teal')
        ax.legend()

        plt.savefig(folder+'J_evolution.pdf')


def aggregate_different_betas(base_folder, template, alpha, gamma, betas, n_seeds=8, t_max=500000, test_every=100):
    n_betas = len(betas)

    train_energy_blob = np.zeros((n_betas, n_seeds, t_max//test_every+1))
    test_energy_blob = np.zeros((n_betas, n_seeds, t_max // test_every + 1))
    displacement_blob = np.zeros((n_betas, n_seeds, t_max // test_every + 1))
    distance_blob = np.zeros((n_betas, n_seeds, t_max // test_every + 1))

    E_star_test = np.zeros((n_betas, n_seeds))
    E_star_train = np.zeros((n_betas, n_seeds))
    E_init_test = np.zeros((n_betas, n_seeds))
    E_init_train = np.zeros((n_betas, n_seeds))

    for beta_idx, beta in enumerate(betas):
        folder = base_folder + template.format(alpha, gamma, beta)
        for seed in range(n_seeds):
            E_star_train[beta_idx, seed] = np.loadtxt('{}/seed_{}/E_star_train.txt'.format(folder, seed))
            E_star_test[beta_idx, seed] = np.loadtxt('{}/seed_{}/E_star_test.txt'.format(folder, seed))
            E_init_train[beta_idx, seed] = np.loadtxt('{}/seed_{}/E_train_init.txt'.format(folder, seed))
            E_init_test[beta_idx, seed] = np.loadtxt('{}/seed_{}/E_test_init.txt'.format(folder, seed))
            train_energy_blob[beta_idx, seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))  - E_star_train[beta_idx, seed]
            test_energy_blob[beta_idx, seed] = np.load('{}/seed_{}/test_energy_acc.npy'.format(folder, seed)) - E_star_test[beta_idx, seed]
            distance_blob[beta_idx, seed] = np.load('{}/seed_{}/distance_to_map_acc.npy'.format(folder, seed))
            displacement_blob[beta_idx, seed] = np.load('{}/seed_{}/distance_to_init_acc.npy'.format(folder, seed))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axes = plt.subplots(1,3, figsize=(15,4))

    ax = axes[0]
    for beta_idx, beta, c_line, c_fill in zip(range(len(betas)), betas, ['r', 'g', 'b', 'm', 'orange'], [PASTEL_RED, PASTEL_GREEN, PASTEL_BLUE, PASTEL_MAGENTA, PASTEL_ORANGE]):
        # plot_mean_std(ax, train_energy_blob[beta_idx, :,1:], axis=0, c_line=c_line, c_fill=c_fill, log_yscale=True, label=r'$\beta={:.1e}$'.format(beta))
        plot_mean_std(ax, train_energy_blob[beta_idx, :,1:], axis=0, c_line=c_line, c_fill=c_fill, log_yscale=True, label=r'$\beta={:.1e}$'.format(beta), rasterized=True)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Metropolis steps')
    # ax.set_ylabel('Train energy')
    ax.set_ylabel(r'$\Delta E^{train}$')



    ax = axes[1]
    for beta_idx, beta, c_line, c_fill in zip(range(len(betas)), betas, ['r', 'g', 'b', 'm', 'orange'], [PASTEL_RED, PASTEL_GREEN, PASTEL_BLUE, PASTEL_MAGENTA, PASTEL_ORANGE]):
        # plot_mean_std(ax, distance_blob[beta_idx, :,1:], axis=0, c_line=c_line, c_fill=c_fill, log_yscale=True, label=r'$\beta={:.1e}$'.format(beta))
        plot_mean_std(ax, distance_blob[beta_idx, :,1:], axis=0, c_line=c_line, c_fill=c_fill, log_yscale=True, label=r'$\beta={:.1e}$'.format(beta), rasterized=True)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Metropolis steps')
    ax.set_ylabel('Distance to MAP')


    ax = axes[2]
    for beta_idx, beta, c_line, c_fill in zip(range(len(betas)), betas, ['r', 'g', 'b', 'm', 'orange'], [PASTEL_RED, PASTEL_GREEN, PASTEL_BLUE, PASTEL_MAGENTA, PASTEL_ORANGE]):
        plot_mean_std(ax, test_energy_blob[beta_idx, :,1:], axis=0, c_line=c_line, c_fill=c_fill, label=r'$\beta={:.1e}$'.format(beta), rasterized=True)

    ax.axhline(y=0., ls='--', c='k')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Metropolis steps')
    ax.set_ylabel(r'$\Delta E^{test}$')

    fig.tight_layout()
    fig.savefig('{}/varying_temperature_alpha_{}_gamma_{}.pdf'.format(base_folder, alpha, gamma))
    plt.close()


if __name__ == '__main__':
    # params = {
    #         # Simulation parameters
    #         'n_neurons': 100,
    #         'alpha': 3.,
    #         'gamma': 1.,
    #         'beta_normalized': 10.,
    #         't_max': 500000,
    #         # 'change_size': .1,
    #         'change_size': .1,
    #         'n_seeds': 8,
    #
    #         # method used to select reference C
    #         'which_C': 'random',
    #
    #         # Multi-threading and IO params
    #         'n_threads': 8,
    #         'silent': False,
    #     'test_every': 100,
    #     }
    #
    #
    #
    #
    #
    # # n = 2
    # # tmax = 10000
    # # test_every = 5
    # # start_from = 'random'
    # #
    # # params_to_vary = {
    # #                     'alpha' : [2., 50., ],
    # #                     'gamma': [1e-2, 1e-1],
    # #                     'beta': [1e6, 1e4],
    # #                     }
    # #
    # # # change_size = 1.
    # # change_size = .01
    # # params['t_max'] = tmax
    # # params['n_neurons'] = n
    # # params['test_every'] = test_every
    # #
    # #
    # #
    # # base_out_dir = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    # # for alpha in params_to_vary['alpha']:
    # #     for gamma in params_to_vary['gamma']:
    # #         for beta in params_to_vary['beta']:
    # #             out_dir = base_out_dir + 'alpha_{}_gamma_{}_beta_{}/'.format(alpha, gamma, beta)
    # #             make_aggregates(out_dir, t_max=tmax, test_every=test_every)
    # #             plot_J_n_2(out_dir, t_max=tmax, test_every=test_every)
    #
    #
    # n = 10
    # tmax = 100000
    # test_every = 50
    # start_from = 'random'
    #
    # params_to_vary = {
    #                     'alpha' : [2., 50., ],
    #                     'gamma': [1e-2, 1e-1],
    #                     'beta': [1e6, 1e4],
    #                     }
    #
    # # change_size = 1.
    # change_size = .01
    # params['t_max'] = tmax
    # params['n_neurons'] = n
    # params['test_every'] = test_every
    # base_out_dir = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    # for alpha in params_to_vary['alpha']:
    #     for gamma in params_to_vary['gamma']:
    #         for beta in params_to_vary['beta']:
    #             out_dir = base_out_dir + 'alpha_{}_gamma_{}_beta_{}/'.format(alpha, gamma, beta)
    #             make_aggregates(out_dir, t_max=tmax, test_every=test_every)


    start_from = 'random'


    #######################################################################
    #######################################################################
    #######################################################################

    # n = 2
    # tmax = 50000
    # test_every = 5
    # change_size = .05
    # params_to_vary = {
    #                     'alpha' : [.5, 2., 10., 50., ],
    #                     'gamma': [1e-2, 1e2, 1, 1e-1, 1e1],
    #                     'beta': np.exp(np.linspace(*np.log([1e4, 1e8]), 10)),
    #                     }
    #
    # base_folder = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    # template = 'alpha_{}_gamma_{}_beta_{}/'
    # for alpha in params_to_vary['alpha']:
    #     for gamma in params_to_vary['gamma']:
    #         betas = [9999.99999999999, 215443.46900318784, 4641588.833612769, 99999999.99999982]
    #         aggregate_different_betas(base_folder, template, alpha, gamma, betas, n_seeds=8, t_max=tmax, test_every=test_every)


    #######################################################################
    #######################################################################
    #######################################################################

    n = 50
    tmax = 2000000
    test_every = 50
    change_size = .1
    params_to_vary = {
                        # 'alpha' : [10, .5, 2., 50., ],
                        # 'alpha' : [.5, 2., 10., 50., ],
                        'alpha' : [2., 10., 50., ],
                        'gamma': [1e-2, 1e2, 1, 1e-1, 1e1],
                        # 'beta': [1e8, 1e6, 1e4, ],
                        # 'beta': [1e4, ],
                        # 'beta': [1e6, 1e4, 1e5, 1e7, 3e5],
                        'beta': np.exp(np.linspace(*np.log([1e4, 1e8]), 10)),
                        }

    # params_to_vary = {
    #                     'alpha' : [50., 10.],
    #                     'gamma': [1e-2, 1e1, 1e-1, 1, 1e2, ],
    #                     'beta': np.exp(np.linspace(*np.log([1e8, 1e4]), 10)),
    #                     }

    base_out_dir = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    for alpha in params_to_vary['alpha']:
        for gamma in params_to_vary['gamma']:
            for beta in params_to_vary['beta']:
                out_dir = base_out_dir + 'alpha_{}_gamma_{}_beta_{}/'.format(alpha, gamma, beta)
                make_aggregates(out_dir, t_max=tmax, test_every=test_every)
                if n==2:
                    plot_J_n_2(out_dir, t_max=tmax, test_every=test_every)

    base_folder = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    template = 'alpha_{}_gamma_{}_beta_{}/'
    for alpha in params_to_vary['alpha']:
        for gamma in params_to_vary['gamma']:
            # betas = [9999.99999999999, 215443.46900318825, 4641588.833612769, 99999999.99999982]
            # betas = params_to_vary['beta'][[0, 3, 6, 9]]
            betas = params_to_vary['beta'][::-1][[0, 2, 4, 6]]
            aggregate_different_betas(base_folder, template, alpha, gamma, betas, n_seeds=8, t_max=tmax, test_every=test_every)

    #######################################################################
    #######################################################################
    #######################################################################

    # n = 20
    # tmax = 2000000
    # test_every = 50
    # change_size = .1
    #
    # params_to_vary = {
    #                     'alpha' : [5.],
    #                     'gamma': [5., ],
    #                     'beta': [1e4, 1e5, 1e6, 1e7, 1e8],
    #                     }
    #
    # # base_out_dir = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    # # for alpha in params_to_vary['alpha']:
    # #     for gamma in params_to_vary['gamma']:
    # #         for beta in params_to_vary['beta']:
    # #             out_dir = base_out_dir + 'alpha_{}_gamma_{}_beta_{}/'.format(alpha, gamma, beta)
    # #             make_aggregates(out_dir, t_max=tmax, test_every=test_every)
    #
    # base_folder = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)
    # template = 'alpha_{}_gamma_{}_beta_{}/'
    # for alpha in params_to_vary['alpha']:
    #     for gamma in params_to_vary['gamma']:
    #         # betas = [9999.99999999999, 215443.46900318825, 4641588.833612769, 99999999.99999982]
    #         # betas = params_to_vary['beta'][[0, 3, 6, 9]]
    #         betas = params_to_vary['beta']
    #         aggregate_different_betas(base_folder, template, alpha, gamma, betas, n_seeds=8, t_max=tmax, test_every=test_every)
