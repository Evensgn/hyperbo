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
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma


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
    (key, cov_func, n_dim, baseline_params, gp_params_samples, queried_sub_dataset, ac_func, budget, n_bo_gp_params_samples) = run_args

    print('generating blank dataset')
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
    print('run baseline bo')
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

    print('generating blank dataset')
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

    print('run hyperbo+ bo')
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
        n_bo_gp_params_samples=n_bo_gp_params_samples
    )

    print('computing regret')
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

    print('run bo done')
    return baseline_regrets, gamma_regrets


def test_bo(key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func, gp_distribution_params):
    n_dim = list(dataset.values())[0].x.shape[1]

    baseline_params = GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 1.0,
            'noise_variance': 1e-6,
        }
    )

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

    print('generating task list')

    task_list = []
    size_list = []
    q = 0
    for sub_dataset in dataset.values():
        q += 1
        new_key, key = jax.random.split(key)
        for i in range(n_bo_runs):
            task_list.append((key, cov_func, n_dim, baseline_params, gp_params_samples, sub_dataset, ac_func, budget, n_bo_gamma_samples))
        size_list.append((q, sub_dataset.x.shape[0]))
    print('task_list constructed, {}'.format(len(task_list)))
    print('size_list constructed, {}'.format(size_list))

    task_outputs = []
    i = 0
    for task in task_list:
        i += 1
        print('task number {}'.format(i))
        task_outputs.append(run_bo(task))
    # task_outputs = pool.map(run_bo, task_list)
    print('task_outputs computed')

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
    return baseline_regrets_mean, baseline_regrets_std, baseline_regrets_list, \
           gamma_regrets_mean, gamma_regrets_std, gamma_regrets_list


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

    # @jit
    def nll_func(model_params):
        return obj.neg_log_marginal_likelihood(
          mean_func=mean_func,
          cov_func=cov_func,
          params=GPParams(model=model_params, config=init_params.config),
          dataset=dataset,
          warp_func=warp_func)

    inferred_params, init_nll, inferred_nll = model.train_return_loss()

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, retrieve_params(inferred_params, param_keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))

    # assert (init_nll > inferred_nll)

    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def run(key, dataset_func_combined, dataset_func_split, train_id_list, test_id_list, n_workers, kernel_name, cov_func,
        objective, opt_method, budget, n_bo_runs, n_bo_gamma_samples, ac_func_type, gp_fit_maxiter):
    experiment_name = 'test_hyperbo_plus_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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
        dataset = dataset_func_combined(train_id)
        print('Dataset loaded')
        new_key, key = jax.random.split(key)
        gp_params, nll_logs = fit_gp_params(new_key, dataset, cov_func, objective, opt_method, gp_fit_maxiter)
        results['fit_gp_params'][train_id] = {'gp_params': gp_params, 'nll_logs': nll_logs}
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

    # run BO
    results['bo_results'] = {}
    baseline_regrets_mean_list = []
    baseline_regrets_std_list = []
    gamma_regrets_mean_list = []
    gamma_regrets_std_list = []

    for test_id in test_id_list:
        print('test_id = {}'.format(test_id))
        dataset = dataset_func_combined(test_id)
        print('Dataset loaded')
        time_0 = time.time()
        new_key, key = jax.random.split(key)
        baseline_regrets_mean, baseline_regrets_std, baseline_regrets_list, gamma_regrets_mean, gamma_regrets_std, gamma_regrets_list = \
            test_bo(new_key, pool, dataset, cov_func, budget, n_bo_runs, n_bo_gamma_samples, ac_func, gp_distribution_params)
        results['bo_results'][test_id] = {
            'baseline_regrets_mean': baseline_regrets_mean,
            'baseline_regrets_std': baseline_regrets_std,
            'baseline_regrets_list': baseline_regrets_std_list,
            'gamma_regrets_mean': gamma_regrets_mean,
            'gamma_regrets_std': gamma_regrets_std,
            'gamma_regrets_list': gamma_regrets_std_list,
        }
        print('baseline_regrets_mean = {}'.format(baseline_regrets_mean))
        print('baseline_regrets_std = {}'.format(baseline_regrets_std))
        print('gamma_regrets_mean = {}'.format(gamma_regrets_mean))
        print('gamma_regrets_std = {}'.format(gamma_regrets_std))
        baseline_regrets_mean_list.append(baseline_regrets_mean)
        baseline_regrets_std_list.append(baseline_regrets_std)
        gamma_regrets_mean_list.append(gamma_regrets_mean)
        gamma_regrets_std_list.append(gamma_regrets_std)
        time_1 = time.time()
        print('Time elapsed for {}: {}'.format(test_id, time_1 - time_0))
        input()

    baseline_regrets_mean_list = jnp.array(baseline_regrets_mean_list)
    baseline_regrets_std_list = jnp.array(baseline_regrets_std_list)
    gamma_regrets_mean_list = jnp.array(gamma_regrets_mean_list)
    gamma_regrets_std_list = jnp.array(gamma_regrets_std_list)
    baseline_regrets_mean_total = jnp.mean(baseline_regrets_mean_list, axis=0)
    baseline_regrets_std_total = jnp.mean(baseline_regrets_std_list, axis=0)
    gamma_regrets_mean_total = jnp.mean(gamma_regrets_mean_list, axis=0)
    gamma_regrets_std_total = jnp.mean(gamma_regrets_std_list, axis=0)
    results['bo_results_total'] = {
        'baseline_regrets_mean': baseline_regrets_mean_total,
        'baseline_regrets_std': baseline_regrets_std_total,
        'gamma_regrets_mean': gamma_regrets_mean_total,
        'gamma_regrets_std': gamma_regrets_std_total
    }

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
        f.write('gp_distribution_params = {}\n'.format(gp_distribution_params))
        f.write('\n')

        for test_id in test_id_list:
            f.write('test_id = {}\n'.format(test_id))
            f.write('baseline_regrets_mean = {}\n'.format(results['bo_results'][test_id]['baseline_regrets_mean']))
            f.write('baseline_regrets_std = {}\n'.format(results['bo_results'][test_id]['baseline_regrets_std']))
            f.write('gamma_regrets_mean = {}\n'.format(results['bo_results'][test_id]['gamma_regrets_mean']))
            f.write('gamma_regrets_std = {}\n'.format(results['bo_results'][test_id]['gamma_regrets_std']))
            f.write('\n')

        f.write('baseline_regrets_mean_total = {}\n'.format(baseline_regrets_mean_total))
        f.write('baseline_regrets_std_total = {}\n'.format(baseline_regrets_std_total))
        f.write('gamma_regrets_mean_total = {}\n'.format(gamma_regrets_mean_total))
        f.write('gamma_regrets_std_total = {}\n'.format(gamma_regrets_std_total))

    # generate plots
    plot.plot_hyperbo_plus(results)

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
    # train_id_list = ['5860', '5906']
    # test_id_list = ['5889']
    # train_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
    # train_id_list = ['6766', '4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6767']
    # test_id_list = ['6794', '7607', '7609', '5889']
    # dataset_func_combined = data.hpob_dataset_v2
    # dataset_func_split = data.hpob_dataset_v3

    train_id_list = [0, 1, 2, 3, 4, 5, 6, 7]
    test_id_list = [8, 9]
    dataset_func_combined = data.hyperbo_plus_synthetic_dataset_combined
    dataset_func_split = data.hyperbo_plus_synthetic_dataset_split

    # test_id_list = ['5889', '6794', '7607', '7609']

    n_workers = 96
    budget = 30
    n_bo_runs = 1
    gp_fit_maxiter = 100
    n_bo_gamma_samples = 100

    key = jax.random.PRNGKey(0)

    for kernel_type in kernel_list:
        for ac_func_type in ['ucb']:
            new_key, key = jax.random.split(key)
            run(new_key, dataset_func_combined, dataset_func_split, train_id_list, test_id_list,
                n_workers, kernel_type[0], kernel_type[1], kernel_type[2],
                kernel_type[3], budget, n_bo_runs, n_bo_gamma_samples, ac_func_type, gp_fit_maxiter)

    print('All done.')

