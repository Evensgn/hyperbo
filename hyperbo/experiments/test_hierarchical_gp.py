import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.bo_utils import bayesopt
from hyperbo.bo_utils import data
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import hierarchical_gp
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
from tensorflow_probability.substrates.jax.distributions import Gamma


DEFAULT_WARP_FUNC = None
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params

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
    n_dim = 1
    n_discrete_points = 10
    n_test_funcs = 96
    budget = 50
    noise_variance = 1e-6
    length_scale = 0.05
    gp_fit_maxiter = 3
    different_domain = 1.0
    num_theta_samples = 20
    n_dataset_thetas = 20
    n_dataset_funcs = 5

    key = jax.random.PRNGKey(0)

    for kernel_name, cov_func, objective, opt_method in kernel_list:
        print('kernel_name: ', kernel_name)
        # infer GP parameters from history functions
        key, _ = jax.random.split(key)

        # discrete domain of test functions
        vx = jax.random.uniform(key, (n_discrete_points, n_dim))
        # discrete domain of dataset functions
        if different_domain:
            vx_dataset = jax.random.uniform(key, (n_discrete_points, n_dim)) * different_domain
        else:
            vx_dataset = vx
        params = GPParams(
            model={
                'constant': 5.,
                'lengthscale': 1.0, # this is just a placeholder, does not actually matter
                'signal_variance': 1.0,
                'noise_variance': noise_variance,
                'higher_params': [0.5, 1.0]
            })

        if cov_func in [
            kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
        ]:
            params.config['mlp_features'] = (8,)
            key, _ = jax.random.split(key)
            bf.init_mlp_with_shape(key, params, vx_dataset.shape)
        elif cov_func == kernel.dot_product_mlp:
            key, _ = jax.random.split(key)
            params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * n_dim))
            params.model['dot_prod_bias'] = 0.
            params.config['mlp_features'] = (8,)
            key, _ = jax.random.split(key)
            bf.init_mlp_with_shape(key, params, vx_dataset.shape)

        mean_func = mean.constant
        logging.info(msg=f'params = {params}')

        key, init_key = jax.random.split(key)
        dataset = [(
                   vx_dataset, hierarchical_gp.sample_from_gp(key, mean_func, cov_func, params, vx_dataset, n_dataset_thetas=n_dataset_thetas, n_dataset_funcs=n_dataset_funcs),
                   'all_data')]
        vy = dataset[0][1]
        for i in range(vy.shape[1]):
            dataset.append((vx_dataset, vy[:, i:i + 1]))

        # minimize nll
        init_params = GPParams(
            model={
                'constant': 5.0,
                'lengthscale': 0.1, # this is just a placeholder, does not actually matter
                'signal_variance': 1.0,
                'noise_variance': noise_variance,
                'higher_params': utils.softplus_warp(jnp.array([2.0, 2.0])) # initial values before warp function
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
                'num_theta_samples': num_theta_samples
            })

        print('ground truth higher params:', params.model['higher_params'])
        print('init higher params:', init_params.model['higher_params'])

        if cov_func in [
            kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
        ]:
            init_params.config['mlp_features'] = (8,)
            key, _ = jax.random.split(key)
            bf.init_mlp_with_shape(key, init_params, vx.shape)
        elif cov_func == kernel.dot_product_mlp:
            key, _ = jax.random.split(key)
            init_params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * 2))
            init_params.model['dot_prod_bias'] = 0.
            init_params.config['mlp_features'] = (8,)
            key, _ = jax.random.split(key)
            bf.init_mlp_with_shape(key, init_params, vx.shape)

        warp_func = DEFAULT_WARP_FUNC

        model = hierarchical_gp.HierarchicalGP(
            dataset=dataset,
            mean_func=mean_func,
            cov_func=cov_func,
            params=init_params,
            warp_func=warp_func)

        model.initialize_params(init_key)


        def nll_func(gpparams, gpwarp_func=None):
            # sample gp params first
            global key
            higher_params = gpparams.model['higher_params']
            gamma_alpha, gamma_beta = higher_params[0], higher_params[1]
            gamma = Gamma(gamma_alpha, gamma_beta)
            key, _ = jax.random.split(key, 2)
            thetas = gamma.sample(num_theta_samples, seed=key)
            objectives = []
            # compute objective value for each theta and take average
            for theta in thetas:
                gpparams.model['lengthscale'] = theta
                # only support nll
                objectives.append(jnp.exp(-obj.neg_log_marginal_likelihood(
                    mean_func=model.mean_func,
                    cov_func=model.cov_func,
                    params=gpparams,
                    dataset=model.dataset,
                    warp_func=gpwarp_func
                )))
            return -jnp.log(jnp.mean(jnp.array(objectives)))


        ground_truth_nll = nll_func(params)
        init_nll = nll_func(init_params, warp_func)

        inferred_params = model.train()

        keys = params.model.keys()
        retrieved_inferred_params = dict(
            zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
        print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))
        # print('inferred higher params:', inferred_params.model['higher_params'])

        inferred_nll = nll_func(inferred_params, warp_func)

        print('init_nll = {}, inferred_nll = {}, ground_truth_nll = {}'.format(init_nll, inferred_nll, ground_truth_nll))

        assert (init_nll > inferred_nll)

    print('All done.')

