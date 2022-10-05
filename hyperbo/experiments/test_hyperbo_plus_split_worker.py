import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.basics import data_utils
from hyperbo.bo_utils import bayesopt
from hyperbo.bo_utils import data
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
import hyperbo.bo_utils.acfun as acfun
import jax
import jax.numpy as jnp
import numpy as np
from pathos.multiprocessing import ProcessingPool
import os
import plot
import datetime
import argparse
import math
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma
from functools import partial
import matplotlib.pyplot as plt
import gc


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


def normal_param_from_thetas(thetas):
    return thetas.mean(), thetas.std()


def gamma_param_from_thetas(thetas):
    a_hat = 0.5 / (math.log(thetas.mean()) - np.log(thetas).mean())
    b_hat = a_hat / thetas.mean()
    return a_hat, b_hat


def gamma_mle_correction(alpha, beta, N):
    corrected_a = alpha - 1/N *(3 * alpha - 2/3 * (alpha / (1 + alpha)) - 4/5 * (alpha / (1 + alpha)**2))
    corrected_b = (N - 1) / N * beta
    return corrected_a, corrected_b


def run_bo(run_args):
    (key, cov_func, n_dim, hyperbo_params, gp_params_samples, fixed_gp_params_samples, queried_sub_dataset, ac_func, budget, n_bo_gp_params_samples, padding_len) = run_args

    placeholder_params = GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 1.0,
            'noise_variance': 1e-6,
        }
    )
    mean_func = mean.constant

    if hyperbo_params:
        print('run hyperbo bo')
        key, _ = jax.random.split(key)
        dataset, sub_dataset_key, _ = data.random(
            key=key,
            mean_func=mean_func,
            cov_func=cov_func,
            params=placeholder_params,
            dim=n_dim,
            n_observed=0,
            n_queries=0,
            n_func_historical=0,
            m_points_historical=0
        )
        key, _ = jax.random.split(key)
        hyperbo_observations, _, _ = bayesopt.run_synthetic(
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            mean_func=mean_func,
            cov_func=cov_func,
            init_params=hyperbo_params,
            warp_func=None,
            ac_func=ac_func,
            iters=budget
        )

    print('run random bo')
    key, _ = jax.random.split(key)
    dataset, sub_dataset_key, _ = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=placeholder_params,
        dim=n_dim,
        n_observed=0,
        n_queries=0,
        n_func_historical=0,
        m_points_historical=0
    )
    key, _ = jax.random.split(key)
    random_observations, _, _ = bayesopt.run_synthetic(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=placeholder_params,
        warp_func=None,
        ac_func=acfun.rand,
        iters=budget
    )

    print('run fixed hierarchical gp bo')
    key, _ = jax.random.split(key)
    dataset, sub_dataset_key, _ = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=placeholder_params,
        dim=n_dim,
        n_observed=0,
        n_queries=0,
        n_func_historical=0,
        m_points_historical=0
    )
    key, _ = jax.random.split(key)
    fixed_observations = bayesopt.run_bo_with_gp_params_samples(
        key=key,
        n_dim=n_dim,
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        gp_params_samples=fixed_gp_params_samples,
        ac_func=ac_func,
        iters=budget,
        n_bo_gp_params_samples=n_bo_gp_params_samples,
        padding_len=padding_len
    )

    print('run hyperbo+ bo')
    key, _ = jax.random.split(key)
    dataset, sub_dataset_key, _ = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=placeholder_params,
        dim=n_dim,
        n_observed=0,
        n_queries=0,
        n_func_historical=0,
        m_points_historical=0
    )
    key, _ = jax.random.split(key)
    gamma_observations = bayesopt.run_bo_with_gp_params_samples(
        key=key,
        n_dim=n_dim,
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        gp_params_samples=gp_params_samples,
        ac_func=ac_func,
        iters=budget,
        n_bo_gp_params_samples=n_bo_gp_params_samples,
        padding_len=padding_len
    )

    print('computing regret')
    # compute regrets
    max_f = jnp.max(queried_sub_dataset.y)

    fixed_regrets = []
    max_y = -jnp.inf
    for y in fixed_observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        fixed_regrets.append(max_f - max_y)

    if hyperbo_params:
        hyperbo_regrets = []
        max_y = -jnp.inf
        for y in hyperbo_observations[1]:
            if y[0] > max_y:
                max_y = y[0]
            hyperbo_regrets.append(max_f - max_y)
    else:
        hyperbo_regrets = None

    random_regrets = []
    max_y = -jnp.inf
    for y in random_observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        random_regrets.append(max_f - max_y)

    gamma_regrets = []
    max_y = -jnp.inf
    for y in gamma_observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        gamma_regrets.append(max_f - max_y)

    gc.collect()
    print('run bo done')
    return fixed_regrets, hyperbo_regrets, random_regrets, gamma_regrets


def test_bo(key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func, gp_distribution_params,
            fixed_gp_distribution_params, hyperbo_params, bo_sub_sample_batch_size):
    n_dim = list(dataset.values())[0].x.shape[1]

    # sub sample each sub dataset for large datasets
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        subkey, dataset, bo_sub_sample_batch_size)
    dataset = next(dataset_iter)

    print('sampling gp params')

    # sample gp params
    constant_mean, constant_sigma = gp_distribution_params['constant']
    constant_normal = Normal(constant_mean, constant_sigma)
    lengthscale_a, lengthscale_b = gp_distribution_params['lengthscale']
    lengthscale_gamma = Gamma(lengthscale_a, lengthscale_b)
    signal_variance_a, signal_variance_b = gp_distribution_params['signal_variance']
    signal_variance_gamma = Gamma(signal_variance_a, signal_variance_b)
    noise_variance_a, noise_variance_b = gp_distribution_params['noise_variance']
    noise_variance_gamma = Gamma(noise_variance_a, noise_variance_b)

    new_key, key = jax.random.split(key)
    constants = constant_normal.sample(budget * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    lengthscales = lengthscale_gamma.sample(budget * n_dim * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    signal_variances = signal_variance_gamma.sample(budget * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    noise_variances = noise_variance_gamma.sample(budget * n_bo_gamma_samples, seed=new_key)
    gp_params_samples = (constants, lengthscales, signal_variances, noise_variances)

    # sample fixed gp params
    fixed_constant_mean, fixed_constant_sigma = fixed_gp_distribution_params['constant']
    fixed_constant_normal = Normal(fixed_constant_mean, fixed_constant_sigma)
    fixed_lengthscale_a, fixed_lengthscale_b = fixed_gp_distribution_params['lengthscale']
    fixed_lengthscale_gamma = Gamma(fixed_lengthscale_a, fixed_lengthscale_b)
    fixed_signal_variance_a, fixed_signal_variance_b = fixed_gp_distribution_params['signal_variance']
    fixed_signal_variance_gamma = Gamma(fixed_signal_variance_a, fixed_signal_variance_b)
    fixed_noise_variance_a, fixed_noise_variance_b = fixed_gp_distribution_params['noise_variance']
    fixed_noise_variance_gamma = Gamma(fixed_noise_variance_a, fixed_noise_variance_b)

    '''
    new_key, key = jax.random.split(key)
    fixed_constants = fixed_constant_normal.sample(budget * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_lengthscales = fixed_lengthscale_gamma.sample(budget * n_dim * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_signal_variances = fixed_signal_variance_gamma.sample(budget * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_noise_variances = fixed_noise_variance_gamma.sample(budget * n_bo_gamma_samples, seed=new_key)
    fixed_gp_params_samples = (fixed_constants, fixed_lengthscales, fixed_signal_variances, fixed_noise_variances)
    '''

    new_key, key = jax.random.split(key)
    fixed_constants = fixed_constant_normal.sample(n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_lengthscales = fixed_lengthscale_gamma.sample(n_dim * n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_signal_variances = fixed_signal_variance_gamma.sample(n_bo_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    fixed_noise_variances = fixed_noise_variance_gamma.sample(n_bo_gamma_samples, seed=new_key)
    fixed_gp_params_samples = (fixed_constants, fixed_lengthscales, fixed_signal_variances, fixed_noise_variances)

    print('generating task list')

    task_list = []
    size_list = []
    q = 0
    for sub_dataset in dataset.values():
        q += 1
        new_key, key = jax.random.split(key)
        for i in range(n_bo_runs):
            task_list.append((key, cov_func, n_dim, hyperbo_params, gp_params_samples, fixed_gp_params_samples, sub_dataset, ac_func, budget, n_bo_gamma_samples, bo_sub_sample_batch_size))
        size_list.append((q, sub_dataset.x.shape[0]))
    print('task_list constructed, {}'.format(len(task_list)))
    print('size_list constructed, {}'.format(size_list))

    task_outputs = []
    i = 0
    for task in task_list:
        i += 1
        print('task number {}'.format(i))
        task_outputs.append(run_bo(task))
    '''
    task_outputs = pool.map(run_bo, task_list)
    '''

    print('task_outputs computed')

    fixed_regrets_list = []
    hyperbo_regrets_list = []
    random_regrets_list = []
    gamma_regrets_list = []
    for task_output in task_outputs:
        fixed_regrets, hyperbo_regrets, random_regrets, gamma_regrets = task_output
        fixed_regrets_list.append(fixed_regrets)
        if hyperbo_params:
            hyperbo_regrets_list.append(hyperbo_regrets)
        random_regrets_list.append(random_regrets)
        gamma_regrets_list.append(gamma_regrets)
    fixed_regrets_list = jnp.array(fixed_regrets_list)
    fixed_regrets_mean = jnp.mean(fixed_regrets_list, axis=0)
    fixed_regrets_std = jnp.std(fixed_regrets_list, axis=0)
    if hyperbo_params:
        hyperbo_regrets_list = jnp.array(hyperbo_regrets_list)
        hyperbo_regrets_mean = jnp.mean(hyperbo_regrets_list, axis=0)
        hyperbo_regrets_std = jnp.std(hyperbo_regrets_list, axis=0)
    else:
        hyperbo_regrets_list = None
        hyperbo_regrets_mean = None
        hyperbo_regrets_std = None
    random_regrets_list = jnp.array(random_regrets_list)
    random_regrets_mean = jnp.mean(random_regrets_list, axis=0)
    random_regrets_std = jnp.std(random_regrets_list, axis=0)
    gamma_regrets_list = jnp.array(gamma_regrets_list)
    gamma_regrets_mean = jnp.mean(gamma_regrets_list, axis=0)
    gamma_regrets_std = jnp.std(gamma_regrets_list, axis=0)
    return fixed_regrets_mean, fixed_regrets_std, fixed_regrets_list, \
           hyperbo_regrets_mean, hyperbo_regrets_std, hyperbo_regrets_list, \
           random_regrets_mean, random_regrets_std, random_regrets_list, \
           gamma_regrets_mean, gamma_regrets_std, gamma_regrets_list


def nll_on_dataset(gp_params, mean_func, cov_func, dataset):
    return obj.nll(
        mean_func,
        cov_func,
        gp_params,
        dataset,
        warp_func=None,
        exclude_aligned=True
    )


def gp_nll_sub_dataset_level(key, dataset, cov_func, gp_params, eval_nll_batch_size, eval_nll_n_batches):
    mean_func = mean.constant

    # sub sample each sub dataset for large datasets
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        subkey, dataset, eval_nll_batch_size)

    nll_loss_batches = []
    for i in range(eval_nll_n_batches):
        dataset = next(dataset_iter)
        nll_loss_list = []
        for sub_dataset in dataset.values():
            nll_i = nll_on_dataset(gp_params, mean_func, cov_func, {'only': sub_dataset})
            nll_loss_list.append(nll_i)
        nll_loss = jnp.mean(jnp.array(nll_loss_list))
        nll_loss_batches.append(nll_loss)
        n_sub_dataset = len(nll_loss_list)
    return nll_loss_batches, n_sub_dataset


def hierarchical_gp_nll(key, dataset, cov_func, n_dim, n_nll_gamma_samples, gp_distribution_params,
                        eval_nll_batch_size, eval_nll_n_batches, sub_dataset_level=False):
    mean_func = mean.constant

    # sub sample each sub dataset for large datasets
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        subkey, dataset, eval_nll_batch_size)

    # sample gp params first
    time_0 = time.time()

    constant_mean, constant_sigma = gp_distribution_params['constant']
    constant_normal = Normal(constant_mean, constant_sigma)
    lengthscale_a, lengthscale_b = gp_distribution_params['lengthscale']
    lengthscale_gamma = Gamma(lengthscale_a, lengthscale_b)
    signal_variance_a, signal_variance_b = gp_distribution_params['signal_variance']
    signal_variance_gamma = Gamma(signal_variance_a, signal_variance_b)
    noise_variance_a, noise_variance_b = gp_distribution_params['noise_variance']
    noise_variance_gamma = Gamma(noise_variance_a, noise_variance_b)

    new_key, key = jax.random.split(key)
    constants = constant_normal.sample(n_nll_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    lengthscales = lengthscale_gamma.sample(n_dim * n_nll_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    signal_variances = signal_variance_gamma.sample(n_nll_gamma_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    noise_variances = noise_variance_gamma.sample(n_nll_gamma_samples, seed=new_key)

    time_1 = time.time()

    nll_loss_batches = []
    for i in range(eval_nll_n_batches):
        dataset = next(dataset_iter)
        if sub_dataset_level:
            nll_loss_list = []
            for sub_dataset in dataset.values():
                objectives = []
                for i in range(n_nll_gamma_samples):
                    params_sample = defs.GPParams(
                        model={
                            'constant': constants[i],
                            'lengthscale': lengthscales[i * n_dim:(i + 1) * n_dim],
                            'signal_variance': signal_variances[i],
                            'noise_variance': noise_variances[i]
                        }
                    )
                    nll_i = nll_on_dataset(params_sample, mean_func, cov_func, {'only': sub_dataset})
                    if not jnp.isnan(nll_i):
                        objectives.append(nll_i)
                n_samples_used = len(objectives)
                assert n_samples_used > 0
                objectives = jnp.array(objectives)
                nll_loss_sub_dataset = -(jax.scipy.special.logsumexp(-objectives, axis=0) - jnp.log(n_samples_used))
                nll_loss_list.append(nll_loss_sub_dataset)
            nll_loss = jnp.mean(jnp.array(nll_loss_list))
            nll_loss_batches.append(nll_loss)
            n_for_sub_dataset_level = len(nll_loss_list)
        else:
            objectives = []
            for i in range(n_nll_gamma_samples):
                params_sample = defs.GPParams(
                    model={
                        'constant': constants[i],
                        'lengthscale': lengthscales[i*n_dim:(i+1)*n_dim],
                        'signal_variance': signal_variances[i],
                        'noise_variance': noise_variances[i]
                    }
                )
                nll_i = nll_on_dataset(params_sample, mean_func, cov_func, dataset)
                if not jnp.isnan(nll_i):
                    objectives.append(nll_i)
            n_samples_used = len(objectives)
            objectives = jnp.array(objectives)
            nll_loss = -(jax.scipy.special.logsumexp(-objectives, axis=0) - jnp.log(n_samples_used))
            nll_loss_batches.append(nll_loss)
            n_for_sub_dataset_level = None

    time_2 = time.time()
    print('time for sampling gp params: {}'.format(time_1 - time_0))
    print('time for calculating nll: {}'.format(time_2 - time_1))

    gc.collect()

    return nll_loss_batches, n_for_sub_dataset_level


def fit_gp_params(key, dataset, cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size, adam_learning_rate):
    n_dim = list(dataset.values())[0].x.shape[1]

    # minimize nll
    init_params = GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method': opt_method,
            'maxiter': gp_fit_maxiter,
            'logging_interval': 1,
            'objective': objective,
            'batch_size': fit_gp_batch_size,
            'learning_rate': adam_learning_rate
        })

    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      init_params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, init_params, (0, n_dim))
    elif cov_func == kernel.dot_product_mlp:
      key, _ = jax.random.split(key)
      init_params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * 2))
      init_params.model['dot_prod_bias'] = 0.
      init_params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, init_params, (0, n_dim))

    warp_func = DEFAULT_WARP_FUNC
    mean_func = mean.constant

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    init_key, key = jax.random.split(key)

    model.initialize_params(init_key)

    inferred_params, init_nll, inferred_nll = model.train_return_loss()

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, retrieve_params(inferred_params, param_keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))

    # assert (init_nll > inferred_nll)

    nll_logs = (init_nll, inferred_nll)

    gc.collect()

    return retrieved_inferred_params, nll_logs


def split_fit_gp_params_id(dir_path, key, setup, train_id, dataset_func_combined, dataset_func_split,
                           cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size, adam_learning_rate):
    if setup == 'a':
        dataset = dataset_func_combined(train_id)
    elif setup == 'b':
        dataset, _ = dataset_func_split(train_id)  # only use training set
    else:
        raise ValueError('setup = {} not supported'.format(setup))
    new_key, key = jax.random.split(key)
    gp_params, nll_logs = fit_gp_params(new_key, dataset, cov_func, objective, opt_method, gp_fit_maxiter,
                                        fit_gp_batch_size, adam_learning_rate)
    results = {'gp_params': gp_params, 'nll_logs': nll_logs}
    np.save(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)), results)


def split_alpha_mle(dir_path, setup, train_id_list):
    constant_list = []
    lengthscale_list = []
    signal_variance_list = []
    noise_variance_list = []

    results = {}

    for train_id in train_id_list:
        gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)),
                            allow_pickle=True).item()['gp_params']
        constant_list.append(gp_params['constant'])
        lengthscale_list += list(gp_params['lengthscale'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])

    gp_distribution_params = {}
    gp_distribution_params['constant'] = normal_param_from_thetas(np.array(constant_list))
    gp_distribution_params['lengthscale'] = gamma_param_from_thetas(np.array(lengthscale_list))
    gp_distribution_params['signal_variance'] = gamma_param_from_thetas(np.array(signal_variance_list))
    gp_distribution_params['noise_variance'] = gamma_param_from_thetas(np.array(noise_variance_list))

    results['gp_distribution_params'] = gp_distribution_params
    np.save(os.path.join(dir_path, 'split_alpha_mle_setup_{}.npy'.format(setup)), results)


def split_test_bo_setup_a_id(dir_path, key, test_id, dataset_func_combined, cov_func, budget, n_bo_runs,
                             n_bo_gamma_samples, ac_func_type_list, fixed_gp_distribution_params,
                             bo_sub_sample_batch_size):
    results = {}

    # placeholder
    pool = None

    # read gp_distribution_params
    gp_distribution_params = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_a.npy'),
                                     allow_pickle=True).item()['gp_distribution_params']

    for ac_func_type in ac_func_type_list:
        print('ac_func_type = {}'.format(ac_func_type))
        if ac_func_type == 'ucb':
            ac_func = acfun.ucb
        elif ac_func_type == 'ei':
            ac_func = acfun.ei
        elif ac_func_type == 'pi':
            ac_func = acfun.pi
        elif ac_func_type == 'rand':
            ac_func = acfun.rand
        else:
            raise ValueError('Unknown ac_func_type: {}'.format(ac_func_type))

        results[ac_func_type] = {}

        dataset = dataset_func_combined(test_id)

        new_key, key = jax.random.split(key)
        fixed_regrets_mean, fixed_regrets_std, fixed_regrets_list, \
        _, _, _, \
        random_regrets_mean, random_regrets_std, random_regrets_list, \
        gamma_regrets_mean, gamma_regrets_std, gamma_regrets_list = \
            test_bo(new_key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func,
                    gp_distribution_params, fixed_gp_distribution_params, None, bo_sub_sample_batch_size)
        results[ac_func_type][test_id] = {
            'fixed_regrets_mean': fixed_regrets_mean,
            'fixed_regrets_std': fixed_regrets_std,
            'fixed_regrets_list': fixed_regrets_list,
            'random_regrets_mean': random_regrets_mean,
            'random_regrets_std': random_regrets_std,
            'random_regrets_list': random_regrets_list,
            'gamma_regrets_mean': gamma_regrets_mean,
            'gamma_regrets_std': gamma_regrets_std,
            'gamma_regrets_list': gamma_regrets_list,
        }
    np.save(os.path.join(dir_path, 'split_test_bo_setup_a_id_{}.npy'.format(test_id)), results)


def split_eval_nll_setup_a_id(dir_path, key, id, dataset_func_combined, cov_func,
                              fixed_gp_distribution_params, n_nll_gamma_samples, setup_a_nll_sub_dataset_level,
                              eval_nll_batch_size, eval_nll_n_batches):
    # read gp_distribution_params
    gp_distribution_params = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_a.npy'),
                                     allow_pickle=True).item()['gp_distribution_params']

    dataset = dataset_func_combined(id)

    # compute nll
    n_dim = list(dataset.values())[0].x.shape[1]
    new_key, key = jax.random.split(key)
    fixed_nll_on_batches_i, fixed_n_for_sdl = \
        hierarchical_gp_nll(new_key, dataset, cov_func, n_dim, n_nll_gamma_samples,
                            fixed_gp_distribution_params, eval_nll_batch_size, eval_nll_n_batches,
                            sub_dataset_level=setup_a_nll_sub_dataset_level)
    new_key, key = jax.random.split(key)
    gamma_nll_on_batches_i, gamma_n_for_sdl = \
        hierarchical_gp_nll(new_key, dataset, cov_func, n_dim, n_nll_gamma_samples,
                            gp_distribution_params, eval_nll_batch_size, eval_nll_n_batches,
                            sub_dataset_level=setup_a_nll_sub_dataset_level)

    results = {
        'fixed_nll_on_batches_i': fixed_nll_on_batches_i,
        'fixed_n_for_sdl': fixed_n_for_sdl,
        'gamma_nll_on_batches_i': gamma_nll_on_batches_i,
        'gamma_n_for_sdl': gamma_n_for_sdl,
    }
    np.save(os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(id)), results)


def split_test_bo_setup_b_id(dir_path, key, test_id, dataset_func_split, setup_b_id_list, cov_func,
                             budget, n_bo_runs, n_bo_gamma_samples, ac_func_type_list, fixed_gp_distribution_params,
                             bo_sub_sample_batch_size):
    results = {}

    # placeholder
    pool = None

    # read gp_distribution_params
    gp_distribution_params = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_b.npy'),
                                     allow_pickle=True).item()['gp_distribution_params']
    # read hyperbo params
    hyperbo_params_test_id = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(test_id)),
                                     allow_pickle=True).item()['gp_params']
    hyperbo_params_test_id = GPParams(model=hyperbo_params_test_id)

    for ac_func_type in ac_func_type_list:
        if ac_func_type == 'ucb':
            ac_func = acfun.ucb
        elif ac_func_type == 'ei':
            ac_func = acfun.ei
        elif ac_func_type == 'pi':
            ac_func = acfun.pi
        elif ac_func_type == 'rand':
            ac_func = acfun.rand
        else:
            raise ValueError('Unknown ac_func_type: {}'.format(ac_func_type))

        results[ac_func_type] = {}

        for test_id in setup_b_id_list:
            _, dataset = dataset_func_split(test_id) # only use test set

            new_key, key = jax.random.split(key)
            fixed_regrets_mean, fixed_regrets_std, fixed_regrets_list, \
            hyperbo_regrets_mean, hyperbo_regrets_std, hyperbo_regrets_list, \
            random_regrets_mean, random_regrets_std, random_regrets_list, \
            gamma_regrets_mean, gamma_regrets_std, gamma_regrets_list = \
                test_bo(new_key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func,
                        gp_distribution_params, fixed_gp_distribution_params, hyperbo_params_test_id,
                        bo_sub_sample_batch_size)
            results[ac_func_type] = {
                'fixed_regrets_mean': fixed_regrets_mean,
                'fixed_regrets_std': fixed_regrets_std,
                'fixed_regrets_list': fixed_regrets_list,
                'hyperbo_regrets_mean': hyperbo_regrets_mean,
                'hyperbo_regrets_std': hyperbo_regrets_std,
                'hyperbo_regrets_list': hyperbo_regrets_list,
                'random_regrets_mean': random_regrets_mean,
                'random_regrets_std': random_regrets_std,
                'random_regrets_list': random_regrets_list,
                'gamma_regrets_mean': gamma_regrets_mean,
                'gamma_regrets_std': gamma_regrets_std,
                'gamma_regrets_list': gamma_regrets_list,
            }
    np.save(os.path.join(dir_path, 'split_test_bo_setup_b_id_{}.npy'.format(test_id)), results)


def split_eval_nll_setup_b_id(dir_path, key, id, train_or_test, dataset_func_split, cov_func,
                              fixed_gp_distribution_params, n_nll_gamma_samples, eval_nll_batch_size,
                              eval_nll_n_batches):
    if train_or_test == 'train':
        dataset, _ = dataset_func_split(id)  # only use train set
    elif train_or_test == 'test':
        _, dataset = dataset_func_split(id)  # only use test set
    else:
        raise ValueError('Unknown train_or_test: {}'.format(train_or_test))

    # read gp_distribution_params
    gp_distribution_params = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_b.npy'),
                                     allow_pickle=True).item()['gp_distribution_params']
    # read hyperbo params
    hyperbo_params_id = \
        np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(id)),
                allow_pickle=True).item()['gp_params']
    hyperbo_params_id = GPParams(model=hyperbo_params_id)

    # compute nll
    n_dim = list(dataset.values())[0].x.shape[1]
    new_key, key = jax.random.split(key)
    fixed_nll_on_batches_i, fixed_n_for_sdl = \
        hierarchical_gp_nll(new_key, dataset, cov_func, n_dim, n_nll_gamma_samples,
                            fixed_gp_distribution_params, eval_nll_batch_size, eval_nll_n_batches,
                            sub_dataset_level=True)

    new_key, key = jax.random.split(key)
    gamma_nll_on_batches_i, gamma_n_for_sdl = \
        hierarchical_gp_nll(new_key, dataset, cov_func, n_dim, n_nll_gamma_samples,
                            gp_distribution_params, eval_nll_batch_size, eval_nll_n_batches,
                            sub_dataset_level=True)

    new_key, key = jax.random.split(key)
    hyperbo_nll_on_batches_i, hyperbo_n_for_sdl = \
        gp_nll_sub_dataset_level(new_key, dataset, cov_func, hyperbo_params_id, eval_nll_batch_size,
                                 eval_nll_n_batches)

    results = {
        'fixed_nll_on_batches_i': fixed_nll_on_batches_i,
        'fixed_n_for_sdl': fixed_n_for_sdl,
        'gamma_nll_on_batches_i': gamma_nll_on_batches_i,
        'gamma_n_for_sdl': gamma_n_for_sdl,
        'hyperbo_nll_on_batches_i': hyperbo_nll_on_batches_i,
        'hyperbo_n_for_sdl': hyperbo_n_for_sdl
    }
    np.save(os.path.join(dir_path, 'split_eval_nll_setup_b_{}_id_{}.npy'.format(train_or_test, id)), results)


def split_merge(dir_path, key, group_id, extra_info, train_id_list, test_id_list,
                setup_b_id_list, n_workers, kernel_name, cov_func, objective, opt_method, budget, n_bo_runs,
                n_bo_gamma_samples, ac_func_type_list, gp_fit_maxiter, fixed_gp_distribution_params,
                n_nll_gamma_samples, setup_a_nll_sub_dataset_level, fit_gp_batch_size, bo_sub_sample_batch_size,
                adam_learning_rate, eval_nll_batch_size, eval_nll_n_batches):
    experiment_name = 'test_hyperbo_plus_split_group_id_{}_merge'.format(group_id)
    results = {}

    results['experiment_name'] = experiment_name
    results['extra_info'] = extra_info
    results['train_id_list'] = train_id_list
    results['test_id_list'] = test_id_list
    results['setup_b_id_list'] = setup_b_id_list
    results['n_workers'] = n_workers
    results['kernel_name'] = kernel_name
    results['cov_func'] = cov_func
    results['objective'] = objective
    results['opt_method'] = opt_method
    results['budget'] = budget
    results['n_bo_runs'] = n_bo_runs
    results['ac_func_type_list'] = ac_func_type_list
    results['gp_fit_maxiter'] = gp_fit_maxiter
    results['fixed_gp_distribution_params'] = fixed_gp_distribution_params
    results['n_nll_gamma_samples'] = n_nll_gamma_samples
    results['setup_a_nll_sub_dataset_level'] = setup_a_nll_sub_dataset_level
    results['fit_gp_batch_size'] = fit_gp_batch_size
    results['bo_sub_sample_batch_size'] = bo_sub_sample_batch_size
    results['adam_learning_rate'] = adam_learning_rate
    results['eval_nll_batch_size'] = eval_nll_batch_size
    results['eval_nll_n_batches'] = eval_nll_n_batches

    # setup a

    results['setup_a'] = {}
    results_a = results['setup_a']

    # fit gp parameters

    results_a['fit_gp_params'] = {}
    for train_id in train_id_list:
        results_a['fit_gp_params'][train_id] = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_a_id_{}.npy'.format(train_id)), allow_pickle=True).item()

    results_a['gp_distribution_params'] = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_a.npy'), allow_pickle=True).item()['gp_distribution_params']

    # run BO and compute NLL
    results_a['bo_results'] = {}
    results_a['bo_results_total'] = {}

    for ac_func_type in ac_func_type_list:
        results_a['bo_results'][ac_func_type] = {}

        fixed_regrets_mean_list = []
        fixed_regrets_std_list = []
        fixed_regrets_all_list = []
        random_regrets_mean_list = []
        random_regrets_std_list = []
        random_regrets_all_list = []
        gamma_regrets_mean_list = []
        gamma_regrets_std_list = []
        gamma_regrets_all_list = []

        for test_id in test_id_list:
            results_a['bo_results'][ac_func_type][test_id] = np.load(os.path.join(dir_path, 'split_test_bo_setup_a_id_{}.npy'.format(test_id)), allow_pickle=True).item()[ac_func_type]
            fixed_regrets_mean = results_a['bo_results'][ac_func_type][test_id]['fixed_regrets_mean']
            fixed_regrets_std = results_a['bo_results'][ac_func_type][test_id]['fixed_regrets_std']
            fixed_regrets_list = results_a['bo_results'][ac_func_type][test_id]['fixed_regrets_list']
            random_regrets_mean = results_a['bo_results'][ac_func_type][test_id]['random_regrets_mean']
            random_regrets_std = results_a['bo_results'][ac_func_type][test_id]['random_regrets_std']
            random_regrets_list = results_a['bo_results'][ac_func_type][test_id]['random_regrets_list']
            gamma_regrets_mean = results_a['bo_results'][ac_func_type][test_id]['gamma_regrets_mean']
            gamma_regrets_std = results_a['bo_results'][ac_func_type][test_id]['gamma_regrets_std']
            gamma_regrets_list = results_a['bo_results'][ac_func_type][test_id]['gamma_regrets_list']

            fixed_regrets_mean_list.append(fixed_regrets_mean)
            fixed_regrets_std_list.append(fixed_regrets_std)
            fixed_regrets_all_list.append(fixed_regrets_list)
            random_regrets_mean_list.append(random_regrets_mean)
            random_regrets_std_list.append(random_regrets_std)
            random_regrets_all_list.append(random_regrets_list)
            gamma_regrets_mean_list.append(gamma_regrets_mean)
            gamma_regrets_std_list.append(gamma_regrets_std)
            gamma_regrets_all_list.append(gamma_regrets_list)

        fixed_regrets_all_list = jnp.concatenate(fixed_regrets_all_list, axis=0)
        fixed_regrets_mean_total = jnp.mean(fixed_regrets_all_list, axis=0)
        fixed_regrets_std_total = jnp.std(fixed_regrets_all_list, axis=0)
        random_regrets_all_list = jnp.concatenate(random_regrets_all_list, axis=0)
        random_regrets_mean_total = jnp.mean(random_regrets_all_list, axis=0)
        random_regrets_std_total = jnp.std(random_regrets_all_list, axis=0)
        gamma_regrets_all_list = jnp.concatenate(gamma_regrets_all_list, axis=0)
        gamma_regrets_mean_total = jnp.mean(gamma_regrets_all_list, axis=0)
        gamma_regrets_std_total = jnp.std(gamma_regrets_all_list, axis=0)

        results_a['bo_results_total'][ac_func_type] = {
            'fixed_regrets_all_list': fixed_regrets_all_list,
            'fixed_regrets_mean': fixed_regrets_mean_total,
            'fixed_regrets_std': fixed_regrets_std_total,
            'random_regrets_all_list': random_regrets_all_list,
            'random_regrets_mean': random_regrets_mean_total,
            'random_regrets_std': random_regrets_std_total,
            'gamma_regrets_all_list': gamma_regrets_all_list,
            'gamma_regrets_mean': gamma_regrets_mean_total,
            'gamma_regrets_std': gamma_regrets_std_total
        }

    fixed_nll_on_train_list = []
    fixed_n_for_train_sdl_total = 0
    gamma_nll_on_train_list = []
    gamma_n_for_train_sdl_total = 0
    for k in range(eval_nll_n_batches):
        fixed_nll_on_train_list.append([])
        gamma_nll_on_train_list.append([])

    for train_id in train_id_list:
        nll_results_train_id = np.load(os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(train_id)), allow_pickle=True).item()

        fixed_nll_on_train_batches_i, fixed_n_for_train_sdl = nll_results_train_id['fixed_nll_on_batches_i'], nll_results_train_id['fixed_n_for_sdl']
        if setup_a_nll_sub_dataset_level:
            fixed_n_for_train_sdl_total += fixed_n_for_train_sdl
            for k in range(eval_nll_n_batches):
                fixed_nll_on_train_list[k].append(fixed_nll_on_train_batches_i[k] * fixed_n_for_train_sdl)
        else:
            for k in range(eval_nll_n_batches):
                fixed_nll_on_train_list[k].append(fixed_nll_on_train_batches_i[k])
        gamma_nll_on_train_batches_i, gamma_n_for_train_sdl = nll_results_train_id['gamma_nll_on_batches_i'], nll_results_train_id['gamma_n_for_sdl']
        if setup_a_nll_sub_dataset_level:
            gamma_n_for_train_sdl_total += gamma_n_for_train_sdl
            for k in range(eval_nll_n_batches):
                gamma_nll_on_train_list[k].append(gamma_nll_on_train_batches_i[k] * gamma_n_for_train_sdl)
        else:
            for k in range(eval_nll_n_batches):
                gamma_nll_on_train_list[k].append(gamma_nll_on_train_batches_i[k])

    fixed_nll_on_test_list = []
    fixed_n_for_test_sdl_total = 0
    gamma_nll_on_test_list = []
    gamma_n_for_test_sdl_total = 0
    for k in range(eval_nll_n_batches):
        fixed_nll_on_test_list.append([])
        gamma_nll_on_test_list.append([])

    for test_id in test_id_list:
        nll_results_test_id = np.load(os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(test_id)), allow_pickle=True).item()

        fixed_nll_on_test_batches_i, fixed_n_for_test_sdl = nll_results_test_id['fixed_nll_on_batches_i'], nll_results_test_id['fixed_n_for_sdl']
        if setup_a_nll_sub_dataset_level:
            fixed_n_for_test_sdl_total += fixed_n_for_test_sdl
            for k in range(eval_nll_n_batches):
                fixed_nll_on_test_list[k].append(fixed_nll_on_test_batches_i[k] * fixed_n_for_test_sdl)
        else:
            for k in range(eval_nll_n_batches):
                fixed_nll_on_test_list[k].append(fixed_nll_on_test_batches_i[k])
        gamma_nll_on_test_batches_i, gamma_n_for_test_sdl = nll_results_test_id['gamma_nll_on_batches_i'], nll_results_test_id['gamma_n_for_sdl']
        if setup_a_nll_sub_dataset_level:
            gamma_n_for_test_sdl_total += gamma_n_for_test_sdl
            for k in range(eval_nll_n_batches):
                gamma_nll_on_test_list[k].append(gamma_nll_on_test_batches_i[k] * gamma_n_for_test_sdl)
        else:
            for k in range(eval_nll_n_batches):
                gamma_nll_on_test_list[k].append(gamma_nll_on_test_batches_i[k])

    fixed_nll_on_test_batches = []
    gamma_nll_on_test_batches = []
    fixed_nll_on_train_batches = []
    gamma_nll_on_train_batches = []
    for k in range(eval_nll_n_batches):
        if setup_a_nll_sub_dataset_level:
            fixed_nll_on_test_batches.append(np.sum(fixed_nll_on_test_list[k]) / fixed_n_for_test_sdl_total)
            gamma_nll_on_test_batches.append(np.sum(gamma_nll_on_test_list[k]) / gamma_n_for_test_sdl_total)
            fixed_nll_on_train_batches.append(np.sum(fixed_nll_on_train_list[k]) / fixed_n_for_train_sdl_total)
            gamma_nll_on_train_batches.append(np.sum(gamma_nll_on_train_list[k]) / gamma_n_for_train_sdl_total)
        else:
            fixed_nll_on_test_batches.append(np.mean(fixed_nll_on_test_list[k]))
            gamma_nll_on_test_batches.append(np.mean(gamma_nll_on_test_list[k]))
            fixed_nll_on_train_batches.append(np.mean(fixed_nll_on_train_list[k]))
            gamma_nll_on_train_batches.append(np.mean(gamma_nll_on_train_list[k]))

    results_a['nll_results'] = {
        'fixed_nll_on_test_mean': np.mean(fixed_nll_on_test_batches),
        'gamma_nll_on_test_mean': np.mean(gamma_nll_on_test_batches),
        'fixed_nll_on_train_mean': np.mean(fixed_nll_on_train_batches),
        'gamma_nll_on_train_mean': np.mean(gamma_nll_on_train_batches),
        'fixed_nll_on_test_std': np.std(fixed_nll_on_test_batches),
        'gamma_nll_on_test_std': np.std(gamma_nll_on_test_batches),
        'fixed_nll_on_train_std': np.std(fixed_nll_on_train_batches),
        'gamma_nll_on_train_std': np.std(gamma_nll_on_train_batches)
    }

    # setup b
    results['setup_b'] = {}
    results_b = results['setup_b']

    # fit gp parameters
    results_b['fit_gp_params'] = {}
    for train_id in setup_b_id_list:
        results_b['fit_gp_params'][train_id] = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(train_id)), allow_pickle=True).item()

    results_b['gp_distribution_params'] = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_b.npy'), allow_pickle=True).item()['gp_distribution_params']

    # run BO and compute NLL
    results_b['bo_results'] = {}
    results_b['bo_results_total'] = {}

    for ac_func_type in ac_func_type_list:
        results_b['bo_results'][ac_func_type] = {}

        fixed_regrets_mean_list = []
        fixed_regrets_std_list = []
        fixed_regrets_all_list = []
        hyperbo_regrets_mean_list = []
        hyperbo_regrets_std_list = []
        hyperbo_regrets_all_list = []
        random_regrets_mean_list = []
        random_regrets_std_list = []
        random_regrets_all_list = []
        gamma_regrets_mean_list = []
        gamma_regrets_std_list = []
        gamma_regrets_all_list = []

        for test_id in setup_b_id_list:
            results_b['bo_results'][ac_func_type][test_id] = np.load(os.path.join(dir_path, 'split_test_bo_setup_b_id_{}.npy'.format(test_id)), allow_pickle=True).item()[ac_func_type]
            fixed_regrets_mean = results_b['bo_results'][ac_func_type][test_id]['fixed_regrets_mean']
            fixed_regrets_std = results_b['bo_results'][ac_func_type][test_id]['fixed_regrets_std']
            fixed_regrets_list = results_b['bo_results'][ac_func_type][test_id]['fixed_regrets_list']
            hyperbo_regrets_mean = results_b['bo_results'][ac_func_type][test_id]['hyperbo_regrets_mean']
            hyperbo_regrets_std = results_b['bo_results'][ac_func_type][test_id]['hyperbo_regrets_std']
            hyperbo_regrets_list = results_b['bo_results'][ac_func_type][test_id]['hyperbo_regrets_list']
            random_regrets_mean = results_b['bo_results'][ac_func_type][test_id]['random_regrets_mean']
            random_regrets_std = results_b['bo_results'][ac_func_type][test_id]['random_regrets_std']
            random_regrets_list = results_b['bo_results'][ac_func_type][test_id]['random_regrets_list']
            gamma_regrets_mean = results_b['bo_results'][ac_func_type][test_id]['gamma_regrets_mean']
            gamma_regrets_std = results_b['bo_results'][ac_func_type][test_id]['gamma_regrets_std']
            gamma_regrets_list = results_b['bo_results'][ac_func_type][test_id]['gamma_regrets_list']

            fixed_regrets_mean_list.append(fixed_regrets_mean)
            fixed_regrets_std_list.append(fixed_regrets_std)
            fixed_regrets_all_list.append(fixed_regrets_list)
            hyperbo_regrets_mean_list.append(hyperbo_regrets_mean)
            hyperbo_regrets_std_list.append(hyperbo_regrets_std)
            hyperbo_regrets_all_list.append(hyperbo_regrets_list)
            random_regrets_mean_list.append(random_regrets_mean)
            random_regrets_std_list.append(random_regrets_std)
            random_regrets_all_list.append(random_regrets_list)
            gamma_regrets_mean_list.append(gamma_regrets_mean)
            gamma_regrets_std_list.append(gamma_regrets_std)
            gamma_regrets_all_list.append(gamma_regrets_list)

        fixed_regrets_all_list = jnp.concatenate(fixed_regrets_all_list, axis=0)
        fixed_regrets_mean_total = jnp.mean(fixed_regrets_all_list, axis=0)
        fixed_regrets_std_total = jnp.std(fixed_regrets_all_list, axis=0)
        hyperbo_regrets_all_list = jnp.concatenate(hyperbo_regrets_all_list, axis=0)
        hyperbo_regrets_mean_total = jnp.mean(hyperbo_regrets_all_list, axis=0)
        hyperbo_regrets_std_total = jnp.std(hyperbo_regrets_all_list, axis=0)
        random_regrets_all_list = jnp.concatenate(random_regrets_all_list, axis=0)
        random_regrets_mean_total = jnp.mean(random_regrets_all_list, axis=0)
        random_regrets_std_total = jnp.std(random_regrets_all_list, axis=0)
        gamma_regrets_all_list = jnp.concatenate(gamma_regrets_all_list, axis=0)
        gamma_regrets_mean_total = jnp.mean(gamma_regrets_all_list, axis=0)
        gamma_regrets_std_total = jnp.std(gamma_regrets_all_list, axis=0)

        results_b['bo_results_total'][ac_func_type] = {
            'fixed_regrets_all_list': fixed_regrets_all_list,
            'fixed_regrets_mean': fixed_regrets_mean_total,
            'fixed_regrets_std': fixed_regrets_std_total,
            'hyperbo_regrets_all_list': hyperbo_regrets_all_list,
            'hyperbo_regrets_mean': hyperbo_regrets_mean_total,
            'hyperbo_regrets_std': hyperbo_regrets_std_total,
            'random_regrets_all_list': random_regrets_all_list,
            'random_regrets_mean': random_regrets_mean_total,
            'random_regrets_std': random_regrets_std_total,
            'gamma_regrets_all_list': gamma_regrets_all_list,
            'gamma_regrets_mean': gamma_regrets_mean_total,
            'gamma_regrets_std': gamma_regrets_std_total
        }

    fixed_nll_on_train_list = []
    fixed_n_for_train_sdl_total = 0
    gamma_nll_on_train_list = []
    gamma_n_for_train_sdl_total = 0
    hyperbo_nll_on_train_list = []
    hyperbo_n_for_train_sdl_total = 0
    for k in range(eval_nll_n_batches):
        fixed_nll_on_train_list.append([])
        gamma_nll_on_train_list.append([])
        hyperbo_nll_on_train_list.append([])

    for train_id in setup_b_id_list:
        nll_results_train_id = np.load(os.path.join(dir_path, 'split_eval_nll_setup_b_train_id_{}.npy'.format(train_id)),
                                       allow_pickle=True).item()
        fixed_nll_on_train_batches_i, fixed_n_for_train_sdl = nll_results_train_id['fixed_nll_on_batches_i'], \
                                                              nll_results_train_id['fixed_n_for_sdl']
        fixed_n_for_train_sdl_total += fixed_n_for_train_sdl
        for k in range(eval_nll_n_batches):
            fixed_nll_on_train_list[k].append(fixed_nll_on_train_batches_i[k] * fixed_n_for_train_sdl)

        gamma_nll_on_train_batches_i, gamma_n_for_train_sdl = nll_results_train_id['gamma_nll_on_batches_i'], \
                                                              nll_results_train_id['gamma_n_for_sdl']
        gamma_n_for_train_sdl_total += gamma_n_for_train_sdl
        for k in range(eval_nll_n_batches):
            gamma_nll_on_train_list[k].append(gamma_nll_on_train_batches_i[k] * gamma_n_for_train_sdl)
        new_key, key = jax.random.split(key)
        hyperbo_nll_on_train_batches_i, hyperbo_n_for_train_sdl = nll_results_train_id['hyperbo_nll_on_batches_i'], \
                                                                  nll_results_train_id['hyperbo_n_for_sdl']
        hyperbo_n_for_train_sdl_total += hyperbo_n_for_train_sdl
        for k in range(eval_nll_n_batches):
            hyperbo_nll_on_train_list[k].append(hyperbo_nll_on_train_batches_i[k] * hyperbo_n_for_train_sdl)

    fixed_nll_on_test_list = []
    fixed_n_for_test_sdl_total = 0
    gamma_nll_on_test_list = []
    gamma_n_for_test_sdl_total = 0
    hyperbo_nll_on_test_list = []
    hyperbo_n_for_test_sdl_total = 0
    for k in range(eval_nll_n_batches):
        fixed_nll_on_test_list.append([])
        gamma_nll_on_test_list.append([])
        hyperbo_nll_on_test_list.append([])

    for test_id in setup_b_id_list:
        nll_results_test_id = np.load(os.path.join(dir_path, 'split_eval_nll_setup_b_test_id_{}.npy'.format(test_id)),
                                      allow_pickle=True).item()

        fixed_nll_on_test_batches_i, fixed_n_for_test_sdl = nll_results_test_id['fixed_nll_on_batches_i'], \
                                                            nll_results_test_id['fixed_n_for_sdl']
        fixed_n_for_test_sdl_total += fixed_n_for_test_sdl
        for k in range(eval_nll_n_batches):
            fixed_nll_on_test_list[k].append(fixed_nll_on_test_batches_i[k] * fixed_n_for_test_sdl)
        gamma_nll_on_test_batches_i, gamma_n_for_test_sdl = nll_results_test_id['gamma_nll_on_batches_i'], \
                                                            nll_results_test_id['gamma_n_for_sdl']
        gamma_n_for_test_sdl_total += gamma_n_for_test_sdl
        for k in range(eval_nll_n_batches):
            gamma_nll_on_test_list[k].append(gamma_nll_on_test_batches_i[k] * gamma_n_for_test_sdl)
        hyperbo_nll_on_test_batches_i, hyperbo_n_for_test_sdl = nll_results_test_id['hyperbo_nll_on_batches_i'], \
                                                                nll_results_test_id['hyperbo_n_for_sdl']
        hyperbo_n_for_test_sdl_total += hyperbo_n_for_test_sdl
        for k in range(eval_nll_n_batches):
            hyperbo_nll_on_test_list[k].append(hyperbo_nll_on_test_batches_i[k] * hyperbo_n_for_test_sdl)

    fixed_nll_on_test_batches = []
    gamma_nll_on_test_batches = []
    hyperbo_nll_on_test_batches = []
    fixed_nll_on_train_batches = []
    gamma_nll_on_train_batches = []
    hyperbo_nll_on_train_batches = []
    for k in range(eval_nll_n_batches):
        fixed_nll_on_test_batches.append(np.sum(fixed_nll_on_test_list[k]) / fixed_n_for_test_sdl_total)
        gamma_nll_on_test_batches.append(np.sum(gamma_nll_on_test_list[k]) / gamma_n_for_test_sdl_total)
        hyperbo_nll_on_test_batches.append(np.sum(hyperbo_nll_on_test_list[k]) / hyperbo_n_for_test_sdl_total)
        fixed_nll_on_train_batches.append(np.sum(fixed_nll_on_train_list[k]) / fixed_n_for_train_sdl_total)
        gamma_nll_on_train_batches.append(np.sum(gamma_nll_on_train_list[k]) / gamma_n_for_train_sdl_total)
        hyperbo_nll_on_train_batches.append(np.sum(hyperbo_nll_on_train_list[k]) / hyperbo_n_for_train_sdl_total)

    results_b['nll_results'] = {
        'fixed_nll_on_test_mean': np.mean(fixed_nll_on_test_batches),
        'gamma_nll_on_test_mean': np.mean(gamma_nll_on_test_batches),
        'hyperbo_nll_on_test_mean': np.mean(hyperbo_nll_on_test_batches),
        'fixed_nll_on_train_mean': np.mean(fixed_nll_on_train_batches),
        'gamma_nll_on_train_mean': np.mean(gamma_nll_on_train_batches),
        'hyperbo_nll_on_train_mean': np.mean(hyperbo_nll_on_train_batches),
        'fixed_nll_on_test_std': np.std(fixed_nll_on_test_batches),
        'gamma_nll_on_test_std': np.std(gamma_nll_on_test_batches),
        'hyperbo_nll_on_test_std': np.std(hyperbo_nll_on_test_batches),
        'fixed_nll_on_train_std': np.std(fixed_nll_on_train_batches),
        'gamma_nll_on_train_std': np.std(gamma_nll_on_train_batches),
        'hyperbo_nll_on_train_std': np.std(hyperbo_nll_on_train_batches)
    }

    # save all results
    dir_path = os.path.join('results', experiment_name)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    np.save(os.path.join(dir_path, 'results.npy'), results)

    # generate plots
    plot.plot_hyperbo_plus(results)

    # write part of results to text file
    with open(os.path.join(dir_path, 'results.txt'), 'w') as f:
        f.write('experiment_name = {}\n'.format(experiment_name))
        f.write('extra_info = {}\n'.format(extra_info))
        f.write('train_id_list = {}\n'.format(train_id_list))
        f.write('test_id_list = {}\n'.format(test_id_list))
        f.write('n_workers = {}\n'.format(n_workers))
        f.write('kernel_name = {}\n'.format(kernel_name))
        f.write('cov_func = {}\n'.format(cov_func))
        f.write('objective = {}\n'.format(objective))
        f.write('opt_method = {}\n'.format(opt_method))
        f.write('budget = {}\n'.format(budget))
        f.write('n_bo_runs = {}\n'.format(n_bo_runs))
        f.write('ac_func_type_list = {}\n'.format(ac_func_type_list))
        f.write('gp_fit_maxiter = {}\n'.format(gp_fit_maxiter))
        f.write('n_bo_gamma_samples = {}\n'.format(n_bo_gamma_samples))
        f.write('n_nll_gamma_samples = {}\n'.format(n_nll_gamma_samples))
        f.write('setup_a_nll_sub_dataset_level = {}\n'.format(setup_a_nll_sub_dataset_level))
        f.write('fit_gp_batch_size = {}\n'.format(fit_gp_batch_size))
        f.write('bo_sub_sample_batch_size = {}\n'.format(bo_sub_sample_batch_size))
        f.write('adam_learning_rate = {}\n'.format(adam_learning_rate))
        f.write('eval_nll_batch_size = {}\n'.format(eval_nll_batch_size))
        f.write('eval_nll_n_batches = {}\n'.format(eval_nll_n_batches))

        f.write('\n')
        for train_id in train_id_list:
            f.write('train_id = {}\n'.format(train_id))
            f.write('nll_logs = {}\n'.format(results_a['fit_gp_params'][train_id]['nll_logs']))
            f.write('\n')
        f.write('gp_distribution_params = {}\n'.format(results_a['gp_distribution_params']))
        f.write('\n')

        for ac_func_type in ac_func_type_list:
            f.write('ac_func_type = {}\n'.format(ac_func_type))
            for test_id in test_id_list:
                f.write('test_id = {}\n'.format(test_id))
                f.write('fixed_regrets_mean = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['fixed_regrets_mean']))
                f.write('fixed_regrets_std = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['fixed_regrets_std']))
                f.write('random_regrets_mean = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['random_regrets_mean']))
                f.write('random_regrets_std = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['random_regrets_std']))
                f.write('gamma_regrets_mean = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['gamma_regrets_mean']))
                f.write('gamma_regrets_std = {}\n'.format(
                    results_a['bo_results'][ac_func_type][test_id]['gamma_regrets_std']))
                f.write('\n')
            f.write('fixed_regrets_mean_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['fixed_regrets_mean']))
            f.write('fixed_regrets_std_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['fixed_regrets_std']))
            f.write('random_regrets_mean_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['random_regrets_mean']))
            f.write('random_regrets_std_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['random_regrets_std']))
            f.write('gamma_regrets_mean_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['gamma_regrets_mean']))
            f.write('gamma_regrets_std_total = {}\n'.format(
                results_a['bo_results_total'][ac_func_type]['gamma_regrets_std']))
            f.write('\n')

        f.write('fixed_nll_on_test_mean = {}\n'.format(results_a['nll_results']['fixed_nll_on_test_mean']))
        f.write('gamma_nll_on_test_mean = {}\n'.format(results_a['nll_results']['gamma_nll_on_test_mean']))
        f.write('fixed_nll_on_train_mean = {}\n'.format(results_a['nll_results']['fixed_nll_on_train_mean']))
        f.write('gamma_nll_on_train_mean = {}\n'.format(results_a['nll_results']['gamma_nll_on_train_mean']))
        f.write('fixed_nll_on_test_std = {}\n'.format(results_a['nll_results']['fixed_nll_on_test_std']))
        f.write('gamma_nll_on_test_std = {}\n'.format(results_a['nll_results']['gamma_nll_on_test_std']))
        f.write('fixed_nll_on_train_std = {}\n'.format(results_a['nll_results']['fixed_nll_on_train_std']))
        f.write('gamma_nll_on_train_std = {}\n'.format(results_a['nll_results']['gamma_nll_on_train_std']))

        f.write('\n\n setup b \n')
        for train_id in setup_b_id_list:
            f.write('train_id = {}\n'.format(train_id))
            f.write('nll_logs = {}\n'.format(results_b['fit_gp_params'][train_id]['nll_logs']))
            f.write('\n')
        f.write('gp_distribution_params = {}\n'.format(results_b['gp_distribution_params']))
        f.write('\n')

        for ac_func_type in ac_func_type_list:
            f.write('ac_func_type = {}\n'.format(ac_func_type))
            for test_id in setup_b_id_list:
                f.write('test_id = {}\n'.format(test_id))
                f.write('fixed_regrets_mean = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['fixed_regrets_mean']))
                f.write('fixed_regrets_std = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['fixed_regrets_std']))
                f.write('hyperbo_regrets_mean = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['hyperbo_regrets_mean']))
                f.write('hyperbo_regrets_std = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['hyperbo_regrets_std']))
                f.write('random_regrets_mean = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['random_regrets_mean']))
                f.write('random_regrets_std = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['random_regrets_std']))
                f.write('gamma_regrets_mean = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['gamma_regrets_mean']))
                f.write('gamma_regrets_std = {}\n'.format(
                    results_b['bo_results'][ac_func_type][test_id]['gamma_regrets_std']))
                f.write('\n')
            f.write('fixed_regrets_mean_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['fixed_regrets_mean']))
            f.write('fixed_regrets_std_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['fixed_regrets_std']))
            f.write('hyperbo_regrets_mean_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['hyperbo_regrets_mean']))
            f.write('hyperbo_regrets_std_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['hyperbo_regrets_std']))
            f.write('random_regrets_mean_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['random_regrets_mean']))
            f.write('random_regrets_std_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['random_regrets_std']))
            f.write('gamma_regrets_mean_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['gamma_regrets_mean']))
            f.write('gamma_regrets_std_total = {}\n'.format(
                results_b['bo_results_total'][ac_func_type]['gamma_regrets_std']))
            f.write('\n')

        f.write('fixed_nll_on_test_mean = {}\n'.format(results_b['nll_results']['fixed_nll_on_test_mean']))
        f.write('gamma_nll_on_test_mean = {}\n'.format(results_b['nll_results']['gamma_nll_on_test_mean']))
        f.write('hyperbo_nll_on_test_mean = {}\n'.format(results_b['nll_results']['hyperbo_nll_on_test_mean']))
        f.write('fixed_nll_on_train_mean = {}\n'.format(results_b['nll_results']['fixed_nll_on_train_mean']))
        f.write('gamma_nll_on_train_mean = {}\n'.format(results_b['nll_results']['gamma_nll_on_train_mean']))
        f.write('hyperbo_nll_on_train_mean = {}\n'.format(results_b['nll_results']['hyperbo_nll_on_train_mean']))
        f.write('fixed_nll_on_test_std = {}\n'.format(results_b['nll_results']['fixed_nll_on_test_std']))
        f.write('gamma_nll_on_test_std = {}\n'.format(results_b['nll_results']['gamma_nll_on_test_std']))
        f.write('hyperbo_nll_on_test_std = {}\n'.format(results_b['nll_results']['hyperbo_nll_on_test_std']))
        f.write('fixed_nll_on_train_std = {}\n'.format(results_b['nll_results']['fixed_nll_on_train_std']))
        f.write('gamma_nll_on_train_std = {}\n'.format(results_b['nll_results']['gamma_nll_on_train_std']))
        f.write('hyperbo_nll_on_train_std = {}\n'.format(results_b['nll_results']['hyperbo_nll_on_train_std']))

    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HyperBO+ split.')

    parser.add_argument('--group_id', default='split_0', type=str, help='split group id')
    parser.add_argument('--mode', default='', type=str, help='mode')
    parser.add_argument('--dataset_id', default='', type=str, help='dataset id')
    parser.add_argument('--key_0', default=0, type=int, help='key 0')
    parser.add_argument('--key_1', default=0, type=int, help='key 1')
    args = parser.parse_args()

    dir_path = os.path.join('results', 'test_hyperbo_plus_split', args.group_id)

    # construct the jax random key
    key = jnp.array([args.key_0, args.key_1], dtype=jnp.uint32)

    # read configs
    configs = np.load(os.path.join(dir_path, 'configs.npy'), allow_pickle=True).item()

    train_id_list = configs['train_id_list']
    test_id_list = configs['test_id_list']
    setup_b_id_list = configs['setup_b_id_list']
    dataset_func_combined = configs['dataset_func_combined']
    dataset_func_split = configs['dataset_func_split']
    extra_info = configs['extra_info']
    n_workers = configs['n_workers']
    budget = configs['budget']
    n_bo_runs = configs['n_bo_runs']
    gp_fit_maxiter = configs['gp_fit_maxiter']
    n_bo_gamma_samples = configs['n_bo_gamma_samples']
    n_nll_gamma_samples = configs['n_nll_gamma_samples']
    setup_a_nll_sub_dataset_level = configs['setup_a_nll_sub_dataset_level']
    fit_gp_batch_size = configs['fit_gp_batch_size']
    bo_sub_sample_batch_size = configs['bo_sub_sample_batch_size']
    adam_learning_rate = configs['adam_learning_rate']
    eval_nll_batch_size = configs['eval_nll_batch_size']
    eval_nll_n_batches = configs['eval_nll_n_batches']
    ac_func_type_list = configs['ac_func_type_list']
    fixed_gp_distribution_params = configs['fixed_gp_distribution_params']
    kernel_type = configs['kernel_type']

    kernel_name, cov_func, objective, opt_method = kernel_type

    if args.mode == 'fit_gp_params_setup_a_id':
        split_fit_gp_params_id(dir_path, key, 'a', args.dataset_id, dataset_func_combined, dataset_func_split,
                               cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size, adam_learning_rate)
    elif args.mode == 'fit_gp_params_setup_b_id':
        split_fit_gp_params_id(dir_path, key, 'b', args.dataset_id, dataset_func_combined, dataset_func_split,
                               cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size, adam_learning_rate)
    elif args.mode == 'alpha_mle_setup_a':
        split_alpha_mle(dir_path, 'a', train_id_list)
    elif args.mode == 'alpha_mle_setup_b':
        split_alpha_mle(dir_path, 'b', setup_b_id_list)
    elif args.mode == 'test_bo_setup_a_id':
        split_test_bo_setup_a_id(dir_path, key, args.dataset_id, dataset_func_combined, cov_func, budget, n_bo_runs,
                                 n_bo_gamma_samples, ac_func_type_list, fixed_gp_distribution_params,
                                 bo_sub_sample_batch_size)
    elif args.mode == 'test_bo_setup_b_id':
        split_test_bo_setup_b_id(dir_path, key, args.dataset_id, dataset_func_split, setup_b_id_list, cov_func,
                                 budget, n_bo_runs, n_bo_gamma_samples, ac_func_type_list, fixed_gp_distribution_params,
                                 bo_sub_sample_batch_size)
    elif args.mode == 'eval_nll_setup_a_id':
        split_eval_nll_setup_a_id(dir_path, key, args.dataset_id, dataset_func_combined, cov_func,
                                  fixed_gp_distribution_params, n_nll_gamma_samples, setup_a_nll_sub_dataset_level,
                                  eval_nll_batch_size, eval_nll_n_batches)
    elif args.mode == 'eval_nll_setup_b_train_id':
        split_eval_nll_setup_b_id(dir_path, key, args.dataset_id, 'train', dataset_func_split, cov_func,
                                  fixed_gp_distribution_params, n_nll_gamma_samples, eval_nll_batch_size,
                                  eval_nll_n_batches)
    elif args.mode == 'eval_nll_setup_b_test_id':
        split_eval_nll_setup_b_id(dir_path, key, args.dataset_id, 'test', dataset_func_split, cov_func,
                                  fixed_gp_distribution_params, n_nll_gamma_samples, eval_nll_batch_size,
                                  eval_nll_n_batches)
    elif args.mode == 'merge':
        split_merge(dir_path, key, args.group_id, extra_info, train_id_list, test_id_list,
                    setup_b_id_list, n_workers, kernel_name, cov_func, objective, opt_method, budget, n_bo_runs,
                    n_bo_gamma_samples, ac_func_type_list, gp_fit_maxiter, fixed_gp_distribution_params,
                    n_nll_gamma_samples, setup_a_nll_sub_dataset_level, fit_gp_batch_size, bo_sub_sample_batch_size,
                    adam_learning_rate, eval_nll_batch_size, eval_nll_n_batches)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))

