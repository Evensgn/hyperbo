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


def run_bo(run_args):
    (key, cov_func, gp_params, warp_func, queried_sub_dataset, ac_func, budget) = run_args
    key, _ = jax.random.split(key)
    mean_func = mean.constant
    dataset, sub_dataset_key, _ = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=gp_params,
        dim=n_dim,
        n_observed=0,
        n_queries=0,
        n_func_historical=0,
        m_points_historical=0
    )

    observations, _, _ = bayesopt.run_synthetic(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=gp_params,
        warp_func=warp_func,
        ac_func=ac_func,
        iters=budget
    )

    # compute regrets
    max_f = jnp.max(queried_sub_dataset.y)
    regrets = []
    max_y = -jnp.inf
    for y in observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        regrets.append(max_f - max_y)

    return observations, regrets


def test_bo(key, cov_func, gp_params, warp_func, queried_sub_datasets, ac_func, budget, n_bo_runs):
    global pool
    task_list = []
    keys = jax.random.split(key, len(queried_sub_datasets) * n_bo_runs + 1)
    key_idx = 0
    for queried_sub_dataset in queried_sub_datasets:
        for i in range(n_bo_runs):
            task_list.append((keys[key_idx], cov_func, gp_params, warp_func, queried_sub_dataset, ac_func, budget))
            key_idx += 1
    task_outputs = pool.map(run_bo, task_list)
    regrets_list = []
    for task_output in task_outputs:
        _, regrets = task_output
        regrets_list.append(regrets)
    regrets_list = jnp.array(regrets_list)
    regrets_mean = jnp.mean(regrets_list, axis=0)
    regrets_std = jnp.std(regrets_list, axis=0)
    return regrets_mean, regrets_std


def test_estimated_prior(key, cov_func, lengthscale, noise_variance, objective, opt_method, ac_func, n_dim,
                         n_dataset_funcs, n_discrete_points, n_test_funcs, budget, n_bo_runs, visualize_bo,
                         gp_fit_maxiter):
    # infer GP parameters from history functions
    key, _ = jax.random.split(key)

    vx = jax.random.uniform(key, (n_discrete_points, n_dim))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': lengthscale,
            'signal_variance': 1.0,
            'noise_variance': noise_variance,
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      key, _ = jax.random.split(key)
      params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * n_dim))
      params.model['dot_prod_bias'] = 0.
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)

    mean_func = mean.constant
    logging.info(msg=f'params = {params}')

    key, init_key = jax.random.split(key)
    dataset = [(vx, gp.sample_from_gp(key, mean_func, cov_func, params, vx, num_samples=n_dataset_funcs), 'all_data')]
    vy = dataset[0][1]
    for i in range(vy.shape[1]):
        dataset.append((vx, vy[:, i:i+1]))

    # minimize nll
    init_params = GPParams(
        model={
            'constant': 3.0,
            'lengthscale': 0.,
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

    warp_func = DEFAULT_WARP_FUNC

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    model.initialize_params(init_key)

    def reg(gpparams, gpwarp_func=None):
      return obj.sample_mean_cov_regularizer(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func,
          distance=utils.kl_multivariate_normal)

    def nll_func(gpparams, gpwarp_func=None):
        return obj.neg_log_marginal_likelihood(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func)

    ground_truth_reg = reg(params)
    ground_truth_nll = nll_func(params)
    init_reg = reg(init_params, warp_func)
    init_nll = nll_func(init_params, warp_func)

    inferred_params = model.train()

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    inferred_reg = reg(inferred_params, warp_func)
    inferred_nll = nll_func(inferred_params, warp_func)

    print('init_nll = {}, inferred_nll = {}, ground_truth_nll = {}'.format(init_nll, inferred_nll, ground_truth_nll))
    print('init_reg = {}, inferred_reg = {}, ground_truth_reg = {}'.format(init_reg, inferred_reg, ground_truth_reg))

    assert (init_nll > inferred_nll)
    # assert (init_reg > inferred_reg)

    nll_logs = (init_nll, inferred_nll, ground_truth_nll)
    reg_logs = (init_reg, inferred_reg, ground_truth_reg)

    # compare inferred params with ground truth params
    # generate test functions
    key, _ = jax.random.split(key)
    y_queries = gp.sample_from_gp(key, mean_func, cov_func, params, vx, num_samples=n_test_funcs)
    queried_sub_datasets = []
    for i in range(n_test_funcs):
        queried_sub_datasets.append(defs.SubDataset(x=vx, y=y_queries[:, i:i+1]))

    key_1, key_2, key_3, key = jax.random.split(key, 4)
    results_groundtruth = test_bo(key_1, cov_func, params, None, queried_sub_datasets, ac_func, budget, n_bo_runs)
    results_inferred = test_bo(key_2, cov_func, inferred_params, warp_func, queried_sub_datasets, ac_func, budget,
                               n_bo_runs)
    results_random = test_bo(key_3, cov_func, params, None, queried_sub_datasets, acfun.rand, budget, n_bo_runs)

    if visualize_bo and n_dim == 1:
        n_visualize_grid_points = 100
        grid_points = jnp.linspace(0.0, 1.0, n_visualize_grid_points).reshape(n_visualize_grid_points, 1)
        key, _ = jax.random.split(key)
        f_x = jnp.vstack((grid_points, vx))
        f_y = gp.sample_from_gp(key, mean_func, cov_func, params, f_x)
        queried_sub_dataset = defs.SubDataset(x=vx, y=f_y[n_visualize_grid_points:])
        key_1, key_2, key = jax.random.split(key, 3)
        observations_groundtruth, _ = run_bo((key_1, cov_func, params, None, queried_sub_dataset, ac_func, budget))
        observations_inferred, _ = run_bo((key_2, cov_func, inferred_params, warp_func, queried_sub_dataset, ac_func,
                                           budget))

        null_dataset = {'null': defs.SubDataset(jnp.empty(0), jnp.empty(0))}
        gp_groundtruth = gp.GP(
            dataset=null_dataset,
            mean_func=mean_func,
            cov_func=cov_func,
            params=params,
            warp_func=None
        )
        gp_inferred = gp.GP(
            dataset=null_dataset,
            mean_func=mean_func,
            cov_func=cov_func,
            params=inferred_params,
            warp_func=warp_func
        )
        posterior_list = []
        for i in range(budget):
            only_dataset = {'only': defs.SubDataset(observations_groundtruth[0][:i], observations_groundtruth[1][:i])}
            gp_groundtruth.set_dataset(only_dataset)
            gp_inferred.set_dataset(only_dataset)
            mean_groundtruth, var_groundtruth = gp_groundtruth.predict(f_x, 'only')
            mean_inferred, var_inferred = gp_inferred.predict(f_x, 'only')
            std_groundtruth = jnp.sqrt(var_groundtruth)
            std_inferred = jnp.sqrt(var_inferred)
            posterior_list.append({
                'mean_groundtruth': mean_groundtruth,
                'std_groundtruth': std_groundtruth,
                'mean_inferred': mean_inferred,
                'std_inferred': std_inferred
            })

        visualize_bo_results = {
            'n_visualize_grid_points': n_visualize_grid_points,
            'f_x': f_x,
            'f_y': f_y,
            'posterior_list': posterior_list
        }
    else:
        visualize_bo_results = None

    return results_groundtruth, results_inferred, results_random, nll_logs, reg_logs, params, retrieved_inferred_params, \
           visualize_bo_results


if __name__ == '__main__':
    results = {}

    experiment_name = 'test_estimated_prior_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    n_workers = 96
    n_dim = 2
    n_dataset_funcs = 1
    n_discrete_points = 100
    n_test_funcs = 96
    budget = 50
    n_bo_runs = 3
    visualize_bo = False
    ac_func_type = 'ei'
    noise_variance = 1e-6
    length_scale = 0.05
    gp_fit_maxiter = 100

    '''
    n_workers = 1
    n_dim = 1
    n_dataset_funcs = 1
    n_discrete_points = 100
    n_test_funcs = 1
    budget = 50
    n_bo_runs = 1
    visualize_bo = True
    ac_func_type = 'ucb'
    noise_variance = 1e-6
    length_scale = 0.05
    gp_fit_maxiter = 100
    '''


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

    results['experiment_name'] = experiment_name
    results['n_workers'] = n_workers
    results['n_dim'] = n_dim
    results['n_dataset_funcs'] = n_dataset_funcs
    results['n_discrete_points'] = n_discrete_points
    results['n_test_funcs'] = n_test_funcs
    results['budget'] = budget
    results['n_bo_runs'] = n_bo_runs
    results['visualize_bo'] = visualize_bo
    results['ac_func_type'] = ac_func_type
    results['noise_variance'] = noise_variance
    results['length_scale'] = length_scale

    DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
    GPParams = defs.GPParams
    retrieve_params = params_utils.retrieve_params

    kernel_list = [
        ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
        ('matern32 nll', kernel.matern32, obj.nll, 'lbfgs'),
        ('matern52 nll', kernel.matern52, obj.nll, 'lbfgs'),
        # ('matern32_mlp nll', kernel.matern32_mlp, obj.nll, 'lbfgs'),
        # ('matern52_mlp nll', kernel.matern52_mlp, obj.nll, 'lbfgs'),
        # ('squared_exponential_mlp nll', kernel.squared_exponential_mlp, obj.nll, 'lbfgs'),
        # ('dot_product_mlp nll', kernel.dot_product_mlp, obj.nll, 'lbfgs'),
        # ('dot_product_mlp nll adam', kernel.dot_product_mlp, obj.nll, 'adam'),
        # ('squared_exponential_mlp nll adam', kernel.squared_exponential_mlp, obj.nll, 'adam'),

        ('squared_exponential kl', kernel.squared_exponential, obj.kl, 'lbfgs'),
        ('matern32 kl', kernel.matern32, obj.kl, 'lbfgs'),
        ('matern52 kl', kernel.matern52, obj.kl, 'lbfgs'),
        # ('matern32_mlp kl', kernel.matern32_mlp, obj.kl, 'lbfgs'),
        # ('matern52_mlp kl', kernel.matern52_mlp, obj.kl, 'lbfgs'),
        # ('squared_exponential_mlp kl', kernel.squared_exponential_mlp, obj.kl, 'lbfgs'),
        # ('dot_product_mlp kl', kernel.dot_product_mlp, obj.kl, 'lbfgs'),
        # ('dot_product_mlp kl adam', kernel.dot_product_mlp, obj.kl, 'adam'),
        # ('squared_exponential_mlp kl adam', kernel.squared_exponential_mlp, obj.kl, 'adam')
    ]

    pool = ProcessingPool(nodes=n_workers)

    key = jax.random.PRNGKey(0)

    keys = jax.random.split(key, len(kernel_list) + 1)
    key = keys[-1]

    results['kernel_list'] = kernel_list
    results['kernel_results'] = {}

    for i, kernel_type in enumerate(kernel_list):
        results_groundtruth, results_inferred, results_random, nll_logs, reg_logs, params, retrieved_inferred_params, \
            visualize_bo_results = test_estimated_prior(
            keys[i], kernel_type[1], length_scale, noise_variance, kernel_type[2], kernel_type[3], ac_func, n_dim,
            n_dataset_funcs, n_discrete_points, n_test_funcs, budget, n_bo_runs, visualize_bo, gp_fit_maxiter
        )
        regrets_mean_groundtruth, regrets_std_groundtruth = results_groundtruth
        regrets_mean_inferred, regrets_std_inferred = results_inferred
        regrets_mean_random, regrets_std_random = results_random

        results['kernel_results'][kernel_type[0]] = {
            'regrets_mean_groundtruth': regrets_mean_groundtruth,
            'regrets_std_groundtruth': regrets_std_groundtruth,
            'regrets_mean_inferred': regrets_mean_inferred,
            'regrets_std_inferred': regrets_std_inferred,
            'regrets_mean_random': regrets_mean_random,
            'regrets_std_random': regrets_std_random,
            'nll_logs': nll_logs,
            'reg_logs': reg_logs,
            'params': params,
            'retrieved_inferred_params': retrieved_inferred_params,
            'visualize_bo_results': visualize_bo_results
        }
        print('kernel type {}: regret_mean_groudtruth = {}, regret_mean_inferred = {}, regret_mean_random = {}'.format(
            kernel_type[0],
            regrets_mean_groundtruth[-1],
            regrets_mean_inferred[-1],
            regrets_mean_random[-1] if results_random else None
        ))

    # save results
    dir_path = os.path.join('results', experiment_name)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    np.save(os.path.join(dir_path, 'results.npy'), results)

    with open(os.path.join(dir_path, 'results.txt'), 'w') as f:
        f.write('experiment_name = {}\n'.format(experiment_name))
        f.write('n_workers = {}\n'.format(n_workers))
        f.write('n_dim = {}\n'.format(n_dim))
        f.write('n_dataset_funcs = {}\n'.format(n_dataset_funcs))
        f.write('n_discrete_points = {}\n'.format(n_discrete_points))
        f.write('n_test_funcs = {}\n'.format(n_test_funcs))
        f.write('budget = {}\n'.format(budget))
        f.write('n_bo_runs = {}\n'.format(n_bo_runs))
        f.write('visualize_bo = {}\n'.format(visualize_bo))
        f.write('gp_fit_maxiter = {}\n'.format(gp_fit_maxiter))
        f.write('length_scale = {}\n'.format(length_scale))
        f.write('noise_variance = {}\n'.format(noise_variance))
        f.write('ac_func_type = {}\n'.format(ac_func_type))
        f.write('kernel_list = {}\n'.format(kernel_list))
        f.write('\n')
        for kernel_type in kernel_list:
            f.write('>>> kernel_type = {}:\n'.format(kernel_type[0]))
            f.write('regret_mean_groundtruth = {}\n'.format(results['kernel_results'][kernel_type[0]]['regrets_mean_groundtruth'][-1]))
            f.write('regret_mean_inferred = {}\n'.format(results['kernel_results'][kernel_type[0]]['regrets_mean_inferred'][-1]))
            f.write('regret_mean_random = {}\n'.format(results['kernel_results'][kernel_type[0]]['regrets_mean_random'][-1]))
            f.write('nll_logs = {}\n'.format(results['kernel_results'][kernel_type[0]]['nll_logs']))
            f.write('reg_logs = {}\n'.format(results['kernel_results'][kernel_type[0]]['reg_logs']))
            f.write('params = {}\n'.format(results['kernel_results'][kernel_type[0]]['params']))
            f.write('retrieved_inferred_params = {}\n'.format(results['kernel_results'][kernel_type[0]]['retrieved_inferred_params']))
            f.write('\n')

    # call the plotting code
    plot.plot_estimated_prior(results)

    print('All done.')
