import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
# from hyperbo.bo_utils import bayesopt
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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params

kernel_list = [
    ('squared_exponential nll', kernel.squared_exponential, 'lbfgs'),
    ('matern32 nll', kernel.matern32, 'lbfgs'),
    ('matern52 nll', kernel.matern52, 'lbfgs'),
    ('matern32_mlp nll', kernel.matern32_mlp, 'lbfgs'),
    ('matern52_mlp nll', kernel.matern52_mlp, 'lbfgs'),
    ('squared_exponential_mlp nll', kernel.squared_exponential_mlp, 'lbfgs'),
    ('dot_product_mlp nll', kernel.dot_product_mlp, 'lbfgs'),
    ('squared_exponential euclidean', kernel.squared_exponential, 'lbfgs'),
    ('dot_product_mlp nll adam', kernel.dot_product_mlp, 'adam'),
    ('squared_exponential_mlp nll adam', kernel.squared_exponential_mlp, 'adam')
]


def test_estimated_prior(cov_func, opt_method, ac_func):
    """Test that GP parameters can be inferred correctly."""
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    n = 20
    vx = jax.random.normal(key, (n, 2))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      key, _ = jax.random.split(key)
      params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * 2))
      params.model['dot_prod_bias'] = 0.
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)

    mean_func = mean.constant
    logging.info(msg=f'params = {params}')

    key, init_key = jax.random.split(key)
    dataset = [(vx,
                gp.sample_from_gp(
                    key, mean_func, cov_func, params, vx,
                    num_samples=10), 'all_data')]
    vy = dataset[0][1]
    for i in range(vy.shape[1]):
      dataset.append((vx, vy[:, i:i+1]))

    # Minimize sample_mean_cov_regularizer.
    init_params = GPParams(
        model={
            'constant': 5.1,
            'lengthscale': 0.,
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method':
                opt_method,
            'maxiter':
                5,
            'logging_interval': 1,
            'objective': obj.nll,
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

    logging.info(msg=f'Regularizer on ground truth params = {reg(params)}')
    logging.info(msg=f'NLL on ground truth params = {nll_func(params)}')

    ground_truth_reg = reg(params)
    ground_truth_nll = nll_func(params)

    init_reg = reg(init_params, warp_func)
    init_nll = nll_func(init_params, warp_func)
    logging.info(msg=f'Reg on init params = {init_reg}')
    logging.info(msg=f'NLL on init params = {init_nll}')

    start_time = time.time()
    logging.info(msg=f'init_params={init_params}')
    inferred_params = model.train()
    logging.info(msg=f'Elapsed training time = {time.time() - start_time}')

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    logging.info(msg=f'inferred_params = {retrieved_inferred_params}')

    inferred_reg = reg(inferred_params, warp_func)
    inferred_nll = nll_func(inferred_params, warp_func)
    logging.info(
        msg=f'Reg on inferred params = {inferred_reg} (Before: {init_reg})')
    logging.info(
        msg=f'NLL on inferred params = {inferred_nll} (Before: {init_nll})')

    assert (init_reg > inferred_reg)
    print('init_reg = {}, inferred_reg = {}, ground_truth_reg = {}'.format(init_reg, inferred_reg, ground_truth_reg))
    print('init_nll = {}, inferred_nll = {}, ground_truth_nll = {}'.format(init_nll, inferred_nll, ground_truth_nll))

    '''
    key, init_key = jax.random.split(key)
    dataset, sub_dataset_key, queried_sub_dataset = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        dim=5,
        n_observed=0,
        n_queries=30,
        n_func_historical=2,
        m_points_historical=10)

    logging.info(
        msg=f'dataset: {jax.tree_map(jnp.shape, dataset)}, '
            f'queried sub-dataset key: {sub_dataset_key}'
            f'queried sub-dataset: {jax.tree_map(jnp.shape, queried_sub_dataset)}')

    observations, queries, params_dict = bayesopt.run_synthetic(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=params,
        ac_func=ac_func,
        iters=3)

    print('observations: {}'.format(jax.tree_map(jnp.shape, observations)))
    print('queries: {}'.format(jax.tree_map(jnp.shape, queries)))
    '''


if __name__ == '__main__':
    for kernel_type in kernel_list:
        test_estimated_prior(kernel_type[1], kernel_type[2], acfun.ucb)
        print(kernel_type[0], 'passed')
        # input()
    print('All tests passed')
