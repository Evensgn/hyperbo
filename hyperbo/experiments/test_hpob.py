import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params


def gamma_param_from_thetas(thetas):
    a_hat = 0.5 / (math.log(thetas.mean()) - np.log(thetas).mean())
    b_hat = a_hat / thetas.mean()
    return a_hat, b_hat


def gamma_mle_correction(alpha, beta, N):
    corrected_a = alpha - 1/N *(3 * alpha - 2/3 * (alpha / (1 + alpha)) - 4/5 * (alpha / (1 + alpha)**2))
    corrected_b = (N - 1) / N * beta
    return corrected_a, corrected_b


def run_bo(run_args):
    (key, cov_func, n_dim, baseline_params, gp_gamma_params, queried_sub_dataset, ac_func, budget, n_bo_gamma_samples) = run_args
    key, _ = jax.random.split(key)
    mean_func = mean.constant
    dataset, sub_dataset_key, _ = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=baseline_params,
        dim=n_dim,
        n_observed=0,
        n_queries=0,
        n_func_historical=0,
        m_points_historical=0
    )

    baseline_observations, _, _ = bayesopt.run_synthetic(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=baseline_params,
        warp_func=None,
        ac_func=ac_func,
        iters=budget
    )

    key, _ = jax.random.split(key)
    gamma_observations = bayesopt.run_bo_gamma(
        key=key,
        n_dim=n_dim,
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        gamma_params=gp_gamma_params,
        ac_func=ac_func,
        iters=budget,
        n_bo_gamma_samples=n_bo_gamma_samples
    )

    # compute regrets
    max_f = jnp.max(queried_sub_dataset.y)

    baseline_regrets = []
    max_y = -jnp.inf
    for y in baseline_observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        baseline_regrets.append(max_f - max_y)

    gamma_regrets = []
    max_y = -jnp.inf
    for y in gamma_observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        gamma_regrets.append(max_f - max_y)

    return baseline_regrets, gamma_regrets


def test_bo(key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func, gp_gamma_params):
    n_dim = list(dataset.values())[0].x.shape[1]

    baseline_params = GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 1.0,
            'noise_variance': 1e-6,
        }
    )

    task_list = []
    for sub_dataset in dataset.values():
        new_key, key = jax.random.split(key)
        for i in range(n_bo_runs):
            task_list.append((key, cov_func, n_dim, baseline_params, gp_gamma_params, sub_dataset, ac_func, budget, n_bo_gamma_samples))
    task_outputs = pool.map(run_bo, task_list)

    baseline_regrets_list = []
    gamma_regrets_list = []
    for task_output in task_outputs:
        baseline_regrets, gamma_regrets = task_output
        baseline_regrets_list.append(baseline_regrets)
        gamma_regrets_list.append(gamma_regrets)
    baseline_regrets_list = jnp.array(baseline_regrets_list)
    baseline_regrets_mean = jnp.mean(baseline_regrets_list, axis=0)
    baseline_regrets_std = jnp.std(baseline_regrets_list, axis=0)
    gamma_regrets_list = jnp.array(gamma_regrets_list)
    gamma_regrets_mean = jnp.mean(gamma_regrets_list, axis=0)
    gamma_regrets_std = jnp.std(gamma_regrets_list, axis=0)
    return baseline_regrets_mean, baseline_regrets_std, gamma_regrets_mean, gamma_regrets_std


def fit_gp_params(key, dataset, cov_func, objective, opt_method, gp_fit_maxiter):
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
            'method':
                opt_method,
            'maxiter':
                gp_fit_maxiter,
            'logging_interval': 1,
            'objective': objective,
            'batch_size': 100,
            'learning_rate': 0.001,
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

    def nll_func(gpparams, gpwarp_func=None):
        return obj.neg_log_marginal_likelihood(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func)

    init_nll = nll_func(init_params, warp_func)

    inferred_params = model.train()

    inferred_nll = nll_func(inferred_params, warp_func)

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, retrieve_params(inferred_params, param_keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))

    # assert (init_nll > inferred_nll)

    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def run(key, train_id_list, test_id_list, n_workers, kernel_name, cov_func, objective, opt_method, budget, n_bo_runs,
        n_bo_gamma_samples, ac_func_type, gp_fit_maxiter):
    experiment_name = 'test_hpob_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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

    results = {}

    results['experiment_name'] = experiment_name
    results['train_id_list'] = train_id_list
    results['test_id_list'] = test_id_list
    results['n_workers'] = n_workers
    results['kernel_name'] = kernel_name
    results['cov_func'] = cov_func
    results['objective'] = objective
    results['opt_method'] = opt_method
    results['budget'] = budget
    results['n_bo_runs'] = n_bo_runs
    results['ac_func_type'] = ac_func_type
    results['gp_fit_maxiter'] = gp_fit_maxiter

    pool = ProcessingPool(nodes=n_workers)

    # fit gp parameters
    constant_list = []
    lengthscale_list = []
    signal_variance_list = []
    noise_variance_list = []

    results['fit_gp_params'] = {}
    for train_id in train_id_list:
        print('train_id = {}'.format(train_id))
        dataset = data.hpob_dataset_v2(train_id)
        print('Dataset loaded')
        new_key, key = jax.random.split(key)
        gp_params, nll_logs = fit_gp_params(new_key, dataset, cov_func, objective, opt_method, gp_fit_maxiter)
        results['fit_gp_params'][train_id] = {'gp_params': gp_params, 'nll_logs': nll_logs}
        constant_list.append(gp_params['constant'])
        lengthscale_list += list(gp_params['lengthscale'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])
    
    gp_gamma_params = {}
    gp_gamma_params['constant'] = gamma_param_from_thetas(np.array(constant_list))
    gp_gamma_params['lengthscale'] = gamma_param_from_thetas(np.array(lengthscale_list))
    gp_gamma_params['signal_variance'] = gamma_param_from_thetas(np.array(signal_variance_list))
    gp_gamma_params['noise_variance'] = gamma_param_from_thetas(np.array(noise_variance_list))

    results['gp_gamma_params'] = gp_gamma_params

    # run BO
    results['bo_results'] = {}
    for test_id in test_id_list:
        print('test_id = {}'.format(test_id))
        dataset = data.hpob_dataset_v2(test_id)
        print('Dataset loaded')
        new_key, key = jax.random.split(key)
        baseline_regrets_mean, baseline_regrets_std, gamma_regrets_mean, gamma_regrets_std = \
            test_bo(new_key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func, gp_gamma_params)
        results['bo_results'][test_id] = {
            'baseline_regrets_mean': baseline_regrets_mean,
            'baseline_regrets_std': baseline_regrets_std,
            'gamma_regrets_mean': gamma_regrets_mean,
            'gamma_regrets_std': gamma_regrets_std
        }
        print('baseline_regrets_mean = {}'.format(baseline_regrets_mean))
        print('baseline_regrets_std = {}'.format(baseline_regrets_std))
        print('gamma_regrets_mean = {}'.format(gamma_regrets_mean))
        print('gamma_regrets_std = {}'.format(gamma_regrets_std))

    # save results
    dir_path = os.path.join('results', experiment_name)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    np.save(os.path.join(dir_path, 'results.npy'), results)

    with open(os.path.join(dir_path, 'results.txt'), 'w') as f:
        f.write('experiment_name = {}\n'.format(experiment_name))
        f.write('train_id_list = {}\n'.format(train_id_list))
        f.write('test_id_list = {}\n'.format(test_id_list))
        f.write('n_workers = {}\n'.format(n_workers))
        f.write('kernel_name = {}\n'.format(kernel_name))
        f.write('cov_func = {}\n'.format(cov_func))
        f.write('objective = {}\n'.format(objective))
        f.write('opt_method = {}\n'.format(opt_method))
        f.write('budget = {}\n'.format(budget))
        f.write('n_bo_runs = {}\n'.format(n_bo_runs))
        f.write('ac_func_type = {}\n'.format(ac_func_type))
        f.write('gp_fit_maxiter = {}\n'.format(gp_fit_maxiter))
        f.write('\n')
        for train_id in train_id_list:
            f.write('train_id = {}\n'.format(train_id))
            f.write('nll_logs = {}\n'.format(results['fit_gp_params'][train_id]['nll_logs']))
            f.write('\n')
        f.write('gp_gamma_params = {}\n'.format(gp_gamma_params))

    print('done.')


total_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                 '6794', '7607', '7609', '5889']

kernel_list = [
    # ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
    ('matern32 nll', kernel.matern32, obj.nll, 'lbfgs'),
    # ('matern52 nll', kernel.matern52, obj.nll, 'lbfgs'),
    # ('matern32_mlp nll', kernel.matern32_mlp, obj.nll, 'lbfgs'),
    # ('matern52_mlp nll', kernel.matern52_mlp, obj.nll, 'lbfgs'),
    # ('squared_exponential_mlp nll', kernel.squared_exponential_mlp, obj.nll, 'lbfgs'),
    # ('dot_product_mlp nll', kernel.dot_product_mlp, obj.nll, 'lbfgs'),
    # ('dot_product_mlp nll adam', kernel.dot_product_mlp, obj.nll, 'adam'),
    # ('squared_exponential_mlp nll adam', kernel.squared_exponential_mlp, obj.nll, 'adam'),

    # ('squared_exponential kl', kernel.squared_exponential, obj.kl, 'lbfgs'),
    # ('matern32 kl', kernel.matern32, obj.kl, 'lbfgs'),
    # ('matern52 kl', kernel.matern52, obj.kl, 'lbfgs'),
    # ('matern32_mlp kl', kernel.matern32_mlp, obj.kl, 'lbfgs'),
    # ('matern52_mlp kl', kernel.matern52_mlp, obj.kl, 'lbfgs'),
    # ('squared_exponential_mlp kl', kernel.squared_exponential_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl', kernel.dot_product_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl adam', kernel.dot_product_mlp, obj.kl, 'adam'),
    # ('squared_exponential_mlp kl adam', kernel.squared_exponential_mlp, obj.kl, 'adam')
]


if __name__ == '__main__':
    # train_id_list = ['4796', '5527', '5636', '5859', '5860']
    train_id_list = ['4796', '5860', '5906']
    test_id_list = ['5889']

    n_workers = 96
    budget = 30
    n_bo_runs = 1
    gp_fit_maxiter = 100
    n_bo_gamma_samples = 100

    key = jax.random.PRNGKey(0)

    for kernel_type in kernel_list:
        for ac_func_type in ['ucb']:
            new_key, key = jax.random.split(key)
            run(new_key, train_id_list, test_id_list, n_workers, kernel_type[0], kernel_type[1], kernel_type[2],
                kernel_type[3], budget, n_bo_runs, n_bo_gamma_samples, ac_func_type, gp_fit_maxiter)

    print('All done.')

