import os
import tqdm
import numpy as np
import torch as tch
from gaussian_model import CenteredGM
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.optimize import minimize_scalar as minimize
from multiprocessing import Pool as ThreadPool
from helpers_montecarlo import post_run_parsing, explore_params

from scipy.optimize import brentq
from theory import LikelihoodEstimator
from copy import deepcopy

params = {
        # Simulation parameters
        'n_neurons': 100,
        'alpha': 3.,
        'gamma': 1.,
        'beta_normalized': 10.,
        't_max': 500000,
        # 'change_size': .1,
        'change_size': .1,
        'n_seeds': 8,

        # method used to select reference C
        'which_C': 'random',

        # Multi-threading and IO params
        'n_threads': 8,
        'silent': False,
        'test_every': 100,
        }

# Setting secondary parameters values
params['n_samples'] = int(params['alpha'] * params['n_neurons'])
params['beta'] = params['beta_normalized'] * params['n_neurons'] ** 2



# params_to_vary = {'alpha' : [20., ],
#                     'gamma': [1e-2, 5e-2, 1e-1, .5, 1, 5, 10, 501e2],
#                     'beta_normalized': [1e-3, 1e-1, 1000, 10, 1],
#                     'change_size': [.1],
#
#                     }

params_to_vary = {'alpha' : [2., ],
                    'gamma': [1e-2, 5e-2, 1e-1, .5, 1, 5, 10, 50, 1e2],
                    'beta_normalized': [1e-3, 1e-1, 1000, 10, 1],
                    'change_size': [.1],
                    }


def compute_likelihoods(gaussian_model, J, C, params):
    # Return the three likelihoods, logZ and Q2
    # All of them are at magnitude 1/2, and without the substraction of logZ

    try:
        spectrum, _ = tch.symeig(J, eigenvectors=False)
    except:
        pass


    C_true = tch.from_numpy(gaussian_model.covariance.astype(np.float32))
    N = params['n_neurons']
    gamma = params['gamma']
    alpha = params['alpha']
    start_from = params['start_from']



    def surrogate(mu):
        return 1. - tch.mean(1. / (mu - spectrum)).item()
    mu_opt = brentq(surrogate, spectrum.max().item()+1e-5, spectrum.max().item()+50, maxiter=100)

    mu_gap = deepcopy(mu_opt - spectrum.max().item())

    logZ = .5 * (mu_opt - tch.log(mu_opt-spectrum).mean().item())

    Q2 = tch.sum(spectrum**2) / (2*N)

    L_train = tch.trace(tch.mm(J,C)) / (2*N)


    # Hre, need to do the change of basis for C_true
    L_test = tch.trace(J.mm(C_true)) /(2*N)
    L_gen = L_train - gamma / alpha * Q2

    energy = (0.5 * gamma * Q2 - alpha * L_train + alpha * logZ) #/ N
    energy_test = (0.5 * gamma * Q2 - alpha * L_test + alpha * logZ) #/ N

    return L_train, L_test, L_gen, logZ, Q2, mu_opt, energy, energy_test, mu_gap


def run_one_thread(out_dir, params, seed):
    # Make experiments different
    out_dir += '/seed_{}/'.format(seed)

    # If this fails, the whole exp should fail
    os.makedirs(out_dir, exist_ok=True)

    # Set seeds for reproducibility.
    # WARNING : GPU seeds are kinda weird https://discuss.pytorch.org/t/random-seed-initialization/7854/15
    np.random.seed(seed)
    tch.manual_seed(seed)
    if tch.cuda.is_available(): tch.cuda.manual_seed_all(seed)

    # Parameters unpacking
    N = params['n_neurons']
    alpha = params['alpha']
    t_max = params['t_max']
    beta = params['beta']
    gamma = params['gamma']
    test_every = params['test_every']
    change_size = params['change_size']
    start_from = params['start_from']

    model_to_fit = CenteredGM(N, precision={'scale': .5, 'sparsity': 1.}, seed=seed)

    # Generate two empirical C to check overfitting
    C_train = tch.from_numpy(model_to_fit.get_empirical_C(n_samples= alpha * N).astype(np.float32))
    print('Distance between C train and C true: {}'.format(C_train-model_to_fit.covariance))

    likelihood_estimator = LikelihoodEstimator(model_to_fit, alpha=alpha)
    J_true = likelihood_estimator.J_true
    mu_true = likelihood_estimator.mu_true
    print(C_train.mean(), C_train.std())
    info, J_star = likelihood_estimator.get_j_star(gamma, C_emp=C_train, alpha=alpha)


    mu_star = info['mu']
    Q2_star = info['Q2']
    L_train_star = info['L_train']
    L_test_star = info['L_test']
    logZ_star = info['logZ']

    np.savetxt(out_dir+'J_true.txt', J_true)
    np.savetxt(out_dir+'mu_true.txt', [mu_true])
    np.savetxt(out_dir+'J_star.txt', J_star)
    np.savetxt(out_dir+'mu_star.txt', [mu_star])
    np.savetxt(out_dir+'L_train_star.txt', [L_train_star])
    np.savetxt(out_dir+'L_test_star.txt', [L_test_star])
    np.savetxt(out_dir+'logZ_star.txt', [logZ_star])

    # This incorporates mu !
    print('Distance Jstar to J true: {}'.format(tch.mean((J_true-J_star)**2)))


    L_train_ref, L_test_ref, _, logZ_ref, Q2_ref, mu_ref, ref_energy, ref_energy_test, mu_gap_ref = compute_likelihoods(model_to_fit, J_star, C_train, params)

    np.savetxt(out_dir+'E_star_train.txt', [ref_energy])
    np.savetxt(out_dir+'E_star_test.txt', [ref_energy_test])


    print('comparing between the two methods')
    print('mu : theory {}, montecarlo {}'.format(mu_star, mu_ref))
    print('L_train : theory {}, montecarlo {}'.format(L_train_star, L_train_ref))
    print('L_train : theory {}, montecarlo {}'.format(L_train_star, L_train_ref))
    print('L_test : theory {}, montecarlo {}'.format(L_test_star, L_test_ref))
    print('logZ : theory {}, montecarlo {}'.format(logZ_star, logZ_ref))
    print('Q2 : theory {}, montecarlo {}'.format(Q2_star, Q2_ref))
    print('Energy star: {}'.format(ref_energy))

    if N==2:
        print('J_star: {}'.format(J_star))
        print('c_emp^-1: {}'.format(tch.linalg.inv(C_train)))


    L_train_ref = L_train_ref.numpy()
    # logZ_ref = logZ_ref.numpy()
    L_test_ref = L_test_ref.numpy()
    ref_energy = ref_energy.numpy()
    ref_energy_test = ref_energy_test.numpy()

    # Initialization for J 

    # Rescale the change size so it depends on J star magnitude
    change_size = change_size * np.mean(np.abs(J_star.numpy()))

    if start_from == 'J_star':
        J = deepcopy(J_star.clone())
    elif start_from == 'random':
        J = tch.sqrt(tch.mean(J_star**2)) * tch.randn(*J_star.shape)
        J = .5*(J+J.transpose(0,1))
    elif start_from == 'close_to_J_star':
        J = J_star.clone() * (1.+ .05*tch.randn(*J_star.shape))
        J = .5*(J+J.transpose(0,1))
    else:
        raise RuntimeError('Undefined start condition')

    J_init = deepcopy(J.clone())


    # Initialize the accumulators
    train_energy_acc = np.zeros(t_max//test_every+1)
    test_energy_acc = np.zeros(t_max // test_every + 1)
    mu_acc = np.zeros(t_max//test_every+1)
    mu_gap_acc = np.zeros(t_max//test_every+1)

    L_train_acc = np.zeros(t_max // test_every + 1)
    L_test_acc = np.zeros(t_max // test_every + 1)
    L_gen_acc = np.zeros(t_max // test_every + 1)
    logZ_acc = np.zeros(t_max // test_every + 1)
    Q2_acc = np.zeros(t_max // test_every + 1)

    distance_to_map_acc = np.zeros(t_max // test_every + 1)
    distance_to_init_acc = np.zeros(t_max // test_every + 1)
    accepted_acc = np.zeros(t_max // test_every + 1)


    if N == 2:
        J_acc = np.zeros((t_max // test_every + 1, 4))
        J_eigs_acc = np.zeros((t_max // test_every + 1, 2))

    L_train, L_test, L_gen, logZ, Q2, mu, energy, energy_test, mu_gap = compute_likelihoods(model_to_fit, J, C_train, params)
    L_train_acc[0] = L_train
    L_test_acc[0] = L_test
    L_gen_acc[0] = L_gen
    logZ_acc[0] = logZ
    Q2_acc[0] = Q2
    train_energy_acc[0] = energy
    test_energy_acc[0] = energy_test
    mu_acc[0] = mu
    mu_gap_acc[0] = mu_gap

    distance_to_map_acc[0] = tch.sqrt(tch.mean((J-J_star)**2))
    distance_to_init_acc[0] = tch.sqrt(tch.mean((J-J_init)**2))

    if N == 2:
        plop = J.clone()
        J_acc[0] = np.array([plop[0,0], plop[0,1], plop[1,1], plop[1, 0]])
        j = tch.linalg.eigvalsh(plop)
        J_eigs_acc[0] = j

    current_energy = energy
    accepted_frac = 0

    np.savetxt(out_dir+'J_init.txt', J)
    np.savetxt(out_dir+'mu_init.txt', [mu])
    np.savetxt(out_dir+'E_train_init.txt', [energy])
    np.savetxt(out_dir+'E_test_init.txt', [energy_test])
    np.savetxt(out_dir+'L_train_init.txt', [L_train])
    np.savetxt(out_dir+'L_test_init.txt', [L_test])
    np.savetxt(out_dir+'logZ_init.txt', [logZ])

    print('change_size: {}'.format(change_size))
    # MC loop
    for t in tqdm.tqdm(range(1, t_max)):

        n_repeats = 10

        if N==2:
            n_repeats = 3

        for repeat in range(n_repeats):

            i, j = np.random.randint(N, size=2)
            epsilon = change_size * np.random.randn()
            J_prop = deepcopy(J.clone())
            J_prop[i, j] = J_prop[i, j] + epsilon
            J_prop[j, i] += epsilon

        _, _, _, _, _, _, F_prop, _, _ = compute_likelihoods(model_to_fit, J_prop, C_train, params)

        if np.isnan(F_prop):
            print("Invalid move")
            continue

        delta_F = deepcopy(F_prop - current_energy)

        if delta_F < 0:

            current_energy = deepcopy(F_prop)
            J = J_prop.clone()
            accepted_frac += 1
        elif np.random.rand() < np.exp(-beta*delta_F):
            current_energy = deepcopy(F_prop)
            J = J_prop.clone()
            accepted_frac += 1
        else:
            pass



        # Introduce checkpoints !
        if t % test_every == (test_every - 1):
            idx = t // test_every + 1 # idx 0 is before starting
            L_train, L_test, L_gen, logZ, Q2, mu, current_energy, energy_test, mu_gap = compute_likelihoods(model_to_fit, J, C_train, params)
            train_energy_acc[idx] = current_energy
            test_energy_acc[idx] = energy_test
            L_train_acc[idx] = L_train
            L_test_acc[idx] = L_test
            L_gen_acc[idx] = L_gen
            logZ_acc[idx] = logZ
            Q2_acc[idx] = Q2
            mu_acc[idx] = mu
            mu_gap_acc[idx] = mu_gap

            if N == 2:
                plop = J.clone()
                J_acc[idx] = np.array([plop[0,0], plop[0,1], plop[1,1], plop[1,0]])
                j = tch.linalg.eigvalsh(plop)
                J_eigs_acc[idx] = j


            accepted_acc[idx] = accepted_frac / test_every
            accepted_frac = 0

            distance_to_map_acc[idx] = tch.sqrt(tch.mean((J-J_star)**2))
            distance_to_init_acc[idx] = tch.sqrt(tch.mean((J-J_init)**2))


            if idx % 100 == 0:

                plt.figure()
                plt.plot(train_energy_acc[:idx]-ref_energy)
                plt.axhline(y=0.)
                plt.savefig(out_dir+'delta_train_energy_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(train_energy_acc[:idx])
                plt.axhline(y=ref_energy)
                plt.savefig(out_dir+'train_energy_real_time.png')
                plt.close()


                plt.figure()
                plt.plot(test_energy_acc[:idx]-ref_energy_test)
                plt.axhline(y=0.)
                plt.savefig(out_dir+'delta_test_energy_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(mu_acc[:idx]-mu_star)
                plt.savefig(out_dir+'delta_mu_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(L_train_acc[:idx]-L_train_ref)
                plt.savefig(out_dir+'delta_L_train_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(L_test_acc[:idx]-L_test_ref)
                plt.savefig(out_dir+'delta_L_test_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(L_train_acc[:idx]-logZ_acc[:idx]-L_train_ref+logZ_ref)
                plt.savefig(out_dir+'delta_L_train_with_logZs_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(L_test_acc[:idx]-logZ_acc[:idx]-L_test_ref+logZ_ref)
                plt.savefig(out_dir+'delta_L_test_with_logZs_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(logZ_acc[:idx])
                plt.savefig(out_dir+'logZ_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(distance_to_map_acc[:idx])
                plt.ylim(ymin=0.)
                plt.savefig(out_dir+'distance_to_map_real_time.png')
                plt.close()

                plt.figure()
                plt.plot(accepted_acc[:idx])
                plt.savefig(out_dir+'accepted.png')
                plt.close()

                np.savetxt(out_dir+'J_step_{}.txt'.format(idx), J)

            if idx % 50 == 49:
                np.save(out_dir + 'train_energy_acc_ckpt', train_energy_acc)
                np.save(out_dir + 'mu_acc_ckpt', mu_acc)
                np.save(out_dir + 'L_train_acc_ckpt', L_train_acc)
                np.save(out_dir + 'L_test_acc_ckpt', L_test_acc)
                np.save(out_dir + 'L_gen_acc_ckpt', L_gen_acc)
                np.save(out_dir + 'logZ_acc_ckpt', logZ_acc)
                np.save(out_dir + 'Q2_acc_ckpt', Q2_acc)
                np.save(out_dir + 'distance_to_map_acc_ckpt', distance_to_map_acc)

    np.save(out_dir + 'distance_to_init_acc', distance_to_init_acc)
    np.save(out_dir + 'distance_to_map_acc', distance_to_map_acc)
    np.save(out_dir + 'train_energy_acc', train_energy_acc)
    np.save(out_dir + 'test_energy_acc', test_energy_acc)
    np.save(out_dir + 'L_train_acc', L_train_acc)
    np.save(out_dir + 'L_test_acc', L_test_acc)
    np.save(out_dir + 'L_gen_acc', L_gen_acc)
    np.save(out_dir + 'logZ_acc', logZ_acc)
    np.save(out_dir + 'Q2_acc', Q2_acc)
    np.save(out_dir + 'mu_acc', mu_acc)
    np.save(out_dir + 'mu_gap_acc', mu_gap_acc)


    if N == 2:
        np.save(out_dir + 'J_acc', J_acc)
        np.save(out_dir + 'J_eigs_acc', J_eigs_acc)


if __name__ == '__main__':
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


    #######################################################################
    #######################################################################
    #######################################################################

    # n = 50
    # tmax = 2000000
    # test_every = 50
    # change_size = .1
    # # params_to_vary = {
    # #                     'alpha' : [.5, 2., 10., 50., ],
    # #                     'gamma': [1e-2, 1e2, 1, 1e-1, 1e1],
    # #                     'beta': np.exp(np.linspace(*np.log([1e4, 1e8]), 10)),
    # #                     }
    #
    # # To gain time, start the other way around (almost, started with 10^-2 bc its where n=2 has some hiccups)
    # params_to_vary = {
    #                     'alpha' : [50., 10.],
    #                     'gamma': [1e-2, 1e1, 1e-1, 1, 1e2, ],
    #                     'beta': np.exp(np.linspace(*np.log([1e8, 1e4]), 10)),
    #                     }



    #######################################################################
    #######################################################################
    #######################################################################

    n = 20
    tmax = 2000000
    test_every = 50
    change_size = .1

    params_to_vary = {
                        'alpha' : [5.],
                        'gamma': [5., ],
                        'beta': [1e4, 1e5, 1e6, 1e7, 1e8],
                        }



    #######################################################################
    #######################################################################
    #######################################################################

    params['t_max'] = tmax
    params['n_neurons'] = n
    params['test_every'] = test_every

    base_out_dir = 'out/monte_carlo/n_{}_tmax_{}_change_{}_start_from_{}/'.format(n, tmax, change_size, start_from)

    for alpha in params_to_vary['alpha']:
        for gamma in params_to_vary['gamma']:
            for beta in params_to_vary['beta']:
                tmp_params = deepcopy(params)
                tmp_params['alpha'] = alpha
                tmp_params['gamma'] = gamma
                tmp_params['beta'] = beta
                tmp_params['start_from'] = start_from
                tmp_params['change_size'] = change_size
                out_dir = base_out_dir + 'alpha_{}_gamma_{}_beta_{}'.format(alpha, gamma, beta)

                n_threads = 8
                pool = ThreadPool(n_threads)
                _ = pool.starmap(run_one_thread, zip([out_dir for _ in range(n_threads)], [deepcopy(tmp_params) for _ in range(n_threads)], range(n_threads)))
                # _ = pool.starmap(run_one_thread, zip([out_dir for _ in range(n_threads)], [deepcopy(tmp_params) for _ in range(n_threads)], range(1, 1+n_threads)))
