import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import os
import numpy as np
import pandas as pd
import json

from collections import Mapping

import six

def tuple_reducer(k1, k2):
    if k1 is None:
        return (k2,)
    else:
        return k1 + (k2,)


def path_reducer(k1, k2):
    import os.path
    if k1 is None:
        return k2
    else:
        return os.path.join(k1, k2)

from pathlib2 import PurePath


def tuple_splitter(flat_key):
    return flat_key


def path_splitter(flat_key):
    keys = PurePath(flat_key).parts
    return keys



REDUCER_DICT = {
    'tuple': tuple_reducer,
    'path': path_reducer,
}

SPLITTER_DICT = {
    'tuple': tuple_splitter,
    'path': path_splitter,
}


def flatten(d, reducer='tuple', inverse=False):
    """Flatten dict-like object.

    Parameters
    ----------
    d: dict-like object
        The dict that will be flattened.
    reducer: {'tuple', 'path', function} (default: 'tuple')
        The key joining method. If a function is given, the function will be
        used to reduce.
        'tuple': The resulting key will be tuple of the original keys
        'path': Use ``os.path.join`` to join keys.
    inverse: bool (default: False)
        Whether you want invert the resulting key and value.

    Returns
    -------
    flat_dict: dict
    """
    if isinstance(reducer, str):
        reducer = REDUCER_DICT[reducer]
    flat_dict = {}

    def _flatten(d, parent=None):
        for key, value in six.viewitems(d):
            flat_key = reducer(parent, key)
            if isinstance(value, Mapping):
                _flatten(value, flat_key)
            else:
                if inverse:
                    flat_key, value = value, flat_key
                if flat_key in flat_dict:
                    raise ValueError("duplicated key '{}'".format(flat_key))
                flat_dict[flat_key] = value

    _flatten(d)
    return flat_dict


def nested_set_dict(d, keys, value):
    """Set a value to a sequence of nested keys

    Parameters
    ----------
    d: Mapping
    keys: Sequence[str]
    value: Any
    """
    assert keys
    key = keys[0]
    if len(keys) == 1:
        if key in d:
            raise ValueError("duplicated key '{}'".format(key))
        d[key] = value
        return
    d = d.setdefault(key, {})
    nested_set_dict(d, keys[1:], value)


def unflatten(d, splitter='tuple', inverse=False):
    """Unflatten dict-like object.

    Parameters
    ----------
    d: dict-like object
        The dict that will be unflattened.
    splitter: {'tuple', 'path', function} (default: 'tuple')
        The key splitting method. If a function is given, the function will be
        used to split.
        'tuple': Use each element in the tuple key as the key of the unflattened dict.
        'path': Use ``pathlib.Path.parts`` to split keys.
    inverse: bool (default: False)
        Whether you want to invert the key and value before flattening.

    Returns
    -------
    unflattened_dict: dict
    """
    if isinstance(splitter, str):
        splitter = SPLITTER_DICT[splitter]

    unflattened_dict = {}
    for flat_key, value in six.viewitems(d):
        if inverse:
            flat_key, value = value, flat_key
        key_tuple = splitter(flat_key)
        nested_set_dict(unflattened_dict, key_tuple, value)

    return unflattened_dict


def underscore_splitter(flat_key):
    return flat_key.split('__')

def underscore_reducer(k1, k2):
    if k1 is None:
        return k2
    else:
        return k1 + "__" + k2

def flatten_underscore(normal_dict):
    return flatten(normal_dict, reducer=underscore_reducer)

def unflatten_underscore(normal_dict):
    return unflatten(normal_dict, splitter=underscore_splitter)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def make_index(dir='out/raw'):
    # Explore the raw folder and build a pandas frame with relevant run informations
    hashes = get_immediate_subdirectories(dir)
    # print(dir, os.listdir(dir), hashes)
    all_keys = set()
    dict_of_dicts = {}

    for hash in hashes:
        with open(dir + '/{}/params'.format(hash), 'r') as outfile:
            params = json.load(outfile)

        # Convention : always exchange unflatten dicts, flatten only when needed
        params = flatten_underscore(params)
        dict_of_dicts[hash] = params

    database = pd.DataFrame.from_dict(dict_of_dicts).transpose()
    database.to_pickle(dir + '/parameters_database.pkl')
    database.to_string(open(dir + '/parameters_database_human_readable.txt', mode='w+'))
    database.to_csv(open(dir + '/parameters_database.csv', mode='w+'))
    print('Current index table : {}'.format(database))

def get_siblings(ref_hash, traversal_key, path='out/'):
    # Take the hash of a reference experiment and return list of hashes such that only 'traversal_key' differs
    # If we are working on nested dicts, traversal key should be the underscore joined key

    db = pd.read_pickle(path+'raw/parameters_database.pkl')
    all_keys = list(db.keys())
    hashes = db.index.values.tolist()
    siblings = []
    values = []

    with open(path+'raw/{}/params'.format(ref_hash), 'r') as outfile:
        ref_params = json.load(outfile)

    # Convention : always exchange unflatten dicts, flatten only when needed
    ref_params = flatten_underscore(ref_params)

    for hash in hashes:
        is_sibling = True
        with open(path+'raw/{}/params'.format(hash), 'r') as outfile:
            params = json.load(outfile)

        # Convention : always exchange unflatten dicts, flatten only when needed
        params = flatten_underscore(params)

        for key in all_keys:
            if params[key] != ref_params[key] and key not in [traversal_key, 't_max', 'n_threads', 'n_seeds', 'test_every']:
                is_sibling = False
                break
        if is_sibling:
            siblings.append(hash)
            values.append(params[traversal_key])

    # Correctly sort the directories in increasing values of the param
    sorter = np.argsort(values)

    siblings = np.array(siblings)[sorter]
    values = np.array(values)[sorter]

    print('To vary parameter {} in {}, visit {}'.format(traversal_key, values, siblings))


    return siblings, values

import hashlib
import json
import os
import sys
import numpy as np


import torch
torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Pool as ThreadPool

from itertools import product
from copy import deepcopy



def soft_mkdir(folder, force_new=False):
    if folder[-1] == '/':
        folder = folder[:-1]
    try:
        if force_new:
            raise FileNotFoundError
        with open('{}/.test_out_exists.txt'.format(folder), mode='w+') as f:
            pass
        return folder
    except FileNotFoundError:
        try:
            os.makedirs(folder)
            return folder
        except FileExistsError:
            for i in range(10):
                try:
                    os.makedirs(folder+'_dup{}'.format(i))
                    return folder+'_dup{}'.format(i)
                except FileExistsError:
                    pass
    raise RuntimeError

def underscore_splitter(flat_key):
    return flat_key.split('__')

def underscore_reducer(k1, k2):
    if k1 is None:
        return k2
    else:
        return k1 + "__" + k2

def flatten_underscore(normal_dict):
    return flatten(normal_dict, reducer=underscore_reducer)

def unflatten_underscore(normal_dict):
    return unflatten(normal_dict, splitter=underscore_splitter)

def hash_dict(dict):
    dict = flatten_underscore(dict)
    return hashlib.sha256(json.dumps(dict, sort_keys=True).encode('utf-8')).hexdigest()[:16]

def run_multi_threaded(function, params, out_dir='out/raw/{}'):
    '''
    Run given function with different seeds in an appropriate directory
    :param function: the function to execute with each new set of params; signature (dir:str, params:dict, seed:int) -> None.
                        First step of that function should be to set the seed.
    :param params: the parameters dict to run in multi-threaded (exploration on this done through "explore_params")
    :return: None
    '''
    def get_id_for_dict(in_dict):
        # Transform a parameter dict into a 16 digits hash for easier storage
        # Forget n_seeds and n_threads, if there is no param named like that it will have no effect
        dict_filtered = {key: in_dict[key] for key in in_dict.keys() if key not in ['n_threads', 'n_seeds']}
        return hashlib.sha256(json.dumps(dict_filtered, sort_keys=True).encode('utf-8')).hexdigest()[:16]


    # Convention : always exchange unflatten dicts, flatten only when needed
    params = flatten_underscore(params)

    # Determine the hash for that particular experiment
    hash = get_id_for_dict(params)

    # Convention : always exchange unflatten dicts, flatten only when needed
    params = unflatten_underscore(params)

    # Print a message for debugging
    print('Exp with id {} and params {}'.format(hash, params))
    out_dir = out_dir.format(hash)

    out_dir = soft_mkdir(out_dir, force_new=True)

    with open(out_dir + '/params', 'w') as outfile:
        json.dump(params, outfile)

    pool = ThreadPool(params['n_threads'])

    print([out_dir for _ in range(params['n_seeds'])])
    print(range(int(params['n_seeds'])))

    _ = pool.starmap(function, zip(
            [out_dir for _ in range(params['n_seeds'])],
            [params for _ in range(params['n_seeds'])],
            range(int(params['n_seeds'])))
            )

    with open(out_dir + 'exited_naturally', 'w') as outfile:
        outfile.write('True')


def explore_params(function, base_params, search_grid, out_dir = 'out/raw/{}'):
    '''
    Generate all parameter combinations and run each using run_multi_threaded
    :param base_params: config from which we want to start exploring
    :param search_grid: dict {key: list of values} of values to test for each specified param
    :return:
    '''

    print('Using base configuration {}'.format(base_params))

    # Convention : always exchange unflatten dicts, flatten only when needed
    base_params = flatten_underscore(base_params)
    search_grid = flatten_underscore(search_grid)

    params_to_vary = list(search_grid.keys())
    n_variables = len(params_to_vary)
    n_values_per_param = [len(search_grid[p]) for p in params_to_vary]
    print('Total number of experiments : {}, make sure it is reasonable...'.format(np.prod(n_values_per_param)))
    all_values = list(product(*[search_grid[key] for key in params_to_vary]))

    for param_tuple in all_values:
        tmp = deepcopy(base_params)
        for i in range(n_variables):
            tmp[params_to_vary[i]] = param_tuple[i]

        # Convention : always exchange unflatten dicts, flatten only when needed
        tmp = unflatten_underscore(tmp)

        print('Using variable parameters {}'.format(tmp))
        sys.stdout.flush()
        # Now, call multi-threaded simulation for these params (not optimal, we could start new threads as soon as 1
        # is done, but should be reasonable if n_threads divides n_seeds (if not, might have to wait for one thread
        # to do full simulation before starting the next batch of 12...)
        run_multi_threaded(function, tmp, out_dir = out_dir)





_DEBUG = False

def check_integrity(folder):
    try:
        with open('{}/exited_naturally'.format(folder), 'r') as f:
            pass
    except FileNotFoundError:
        print('Folder {} did not exit naturally, skip it'.format(folder))


def make_average_energy(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    train_energy_blob = np.zeros((n_seeds, t_max//test_every+1))
    test_energy_blob = np.zeros((n_seeds, t_max // test_every + 1))
    distance_blob = np.zeros((n_seeds, t_max // test_every + 1))

    for seed in range(n_seeds):
        train_energy_blob[seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))
        # test_energy_blob[seed] = np.load('{}/seed_{}/test_energy_acc.npy'.format(folder, seed))
        test_energy_blob[seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))
        distance_blob[seed] = np.load('{}/seed_{}/distance_to_map_acc.npy'.format(folder, seed))


    train_mean = np.mean(train_energy_blob, axis=0)[1:]
    train_std = np.std(train_energy_blob, axis=0)[1:]

    test_mean = np.mean(test_energy_blob, axis=0)[1:]
    test_std = np.std(test_energy_blob, axis=0)[1:]

    distance_mean = np.mean(distance_blob, axis=0)[1:]
    distance_std = np.std(distance_blob, axis=0)[1:]


    np.save('{}/E_train.npy'.format(folder), train_energy_blob)
    np.save('{}/E_train_avg.npy'.format(folder), train_mean)
    np.save('{}/E_train_std.npy'.format(folder), train_std)

    np.save('{}/E_test.npy'.format(folder), test_energy_blob)
    np.save('{}/E_test_avg.npy'.format(folder), test_mean)
    np.save('{}/E_test_std.npy'.format(folder), test_std)

    np.save('{}/distance.npy'.format(folder), distance_blob)
    np.save('{}/distance_avg.npy'.format(folder), distance_mean)
    np.save('{}/distance_std.npy'.format(folder), distance_std)

    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), train_mean, yerr=train_std, label='Train')
    plt.errorbar(np.arange(0, t_max, test_every), test_mean, yerr=test_std, label='Test')
    plt.yscale('log')
    plt.legend()
    plt.savefig('{}/energies.png'.format(folder))
    plt.close()

    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), distance_mean, yerr=distance_std)
    plt.yscale('log')
    plt.legend()
    plt.savefig('{}/distances.png'.format(folder))
    plt.close()


def make_average_mu(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    mu_blob = np.zeros((n_seeds, t_max//test_every+1))


    for seed in range(n_seeds):
        mu_blob[seed] = np.load('{}/seed_{}/mu_acc.npy'.format(folder, seed))


    mu_mean = np.mean(mu_blob, axis=0)[1:]
    mu_std = np.std(mu_blob, axis=0)[1:]

    np.save('{}/mus.npy'.format(folder), mu_blob)
    np.save('{}/mu_avg.npy'.format(folder), mu_mean)
    np.save('{}/mu_std.npy'.format(folder), mu_std)

    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), mu_mean, yerr=mu_std)
    plt.savefig('{}/mus.png'.format(folder))
    plt.close()


def summarize_likelihoods(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    L_train_blob = np.zeros((n_seeds, t_max//test_every+1))
    L_test_blob = np.zeros((n_seeds, t_max//test_every+1))
    L_gen_blob = np.zeros((n_seeds, t_max//test_every+1))
    logZ_blob = np.zeros((n_seeds, t_max//test_every+1))
    Q2_blob = np.zeros((n_seeds, t_max//test_every+1))

    for seed in range(n_seeds):
        L_train_blob[seed] = np.load(folder + '/seed_{}/L_train_acc_ckpt.npy'.format(seed))
        L_test_blob[seed] = np.load(folder + '/seed_{}/L_test_acc_ckpt.npy'.format(seed))
        L_gen_blob[seed] = np.load(folder + '/seed_{}/L_gen_acc_ckpt.npy'.format(seed))
        logZ_blob[seed] = np.load(folder + '/seed_{}/logZ_acc_ckpt.npy'.format(seed))
        Q2_blob[seed] = np.load(folder + '/seed_{}/Q2_acc_ckpt.npy'.format(seed))

    np.save(folder + '/L_train', L_train_blob)
    np.save(folder + '/L_test', L_test_blob)
    np.save(folder + '/L_gen', L_gen_blob)
    np.save(folder + '/logZ', logZ_blob)
    np.save(folder + '/Q2', Q2_blob)



def make_eigenvalues_diffmap(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        N = dict['n_neurons']
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    eigenvalues_blob = np.zeros((t_max // test_every + 1, N, n_seeds))
    mu_blob = np.zeros((t_max // test_every + 1, N, n_seeds))
    for seed in range(n_seeds):
        eigenvalues_blob[:, :, seed] = np.load('{}/seed_{}/eigenvalues_acc.npy'.format(folder, seed))
        mu_blob[:, :, seed] = np.repeat(np.load('{}/seed_{}/mu_acc.npy'.format(folder, seed)).reshape(-1, 1), N, axis=1)


    if _DEBUG:
        for t in range(1, 10):
            print(eigenvalues_blob[-t, :, 0].max().item(), eigenvalues_blob[-t, :, 0].min().item(), mu_blob[-t, 0, 0], mu_blob[-t, -1, 0], eigenvalues_blob[-t, :, 0].mean().item())

    np.save('{}/eigenvalues.npy'.format(folder), eigenvalues_blob.reshape((len(eigenvalues_blob), -1)))

    eigenvalues_blob = mu_blob - eigenvalues_blob

    if _DEBUG:
        for t in range(1, 10):
            print(eigenvalues_blob[-t, :, 0].max().item(), eigenvalues_blob[-t, :, 0].min().item(), eigenvalues_blob[-t, :, 0].mean().item())


    # Make a "diffusion map" for the eigenvalues
    n_bins = 1000
    grouping = 10
    v_min, v_max = -5., 5.

    mixed_evs = np.clip(eigenvalues_blob.reshape(eigenvalues_blob.shape[0], -1), v_min, v_max)
    # print(np.min(mixed_evs), np.max(mixed_evs))
    # print(np.sum(mixed_evs == 0), np.sum(mixed_evs == 1.5))
    bounds = [v_min, v_max]
    diff_map = np.zeros((eigenvalues_blob.shape[0] // grouping +1, n_bins))

    for t in range(eigenvalues_blob.shape[0]):
        for ev in mixed_evs[t]:
            diff_map[t // grouping, int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0]))] += 1.

    for t in range(diff_map.shape[0]):
        diff_map[t] /= np.sum(diff_map[t])

    plt.figure()
    plt.imshow(diff_map[-100:].T, origin='lower', extent=[0, t_max, bounds[0], bounds[1]], aspect='auto')
    plt.title(folder.split('/')[-1])
    plt.savefig('{}/eigenvalues_diffmap.png'.format(folder))
    plt.close()



def parse_individual_subfolder(subfolder):
    # This function only acts as a container for other analysis routines (to keep things readable and modular)

    make_average_energy(subfolder)
    summarize_likelihoods(subfolder)
    make_average_mu(subfolder)
    # make_eigenvalues_diffmap(subfolder)



def post_run_parsing(dir='out', sub_dir='montecarlo'):
    make_index(dir + '/' + sub_dir)

    exp_dirs = get_immediate_subdirectories(dir + '/' + sub_dir)
    for exp_dir in exp_dirs:
        print('Treating folder {}'.format(exp_dir))
        parse_individual_subfolder(dir + '/' + sub_dir + '/' + exp_dir)
