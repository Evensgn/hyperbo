# Closed form theta MLE evaluation
# reference https://tminka.github.io/papers/minka-gamma.pdf, note that the beta in the doc is indeed theta so should take beta = 1/theta
# reference for correction: https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation

import math
def gamma_param_from_thetas(thetas, a = None):
  a_hat = 0.5 / (math.log(thetas.mean()) - np.log(thetas).mean())
  if a is not None:
    a_hat = a
    b_hat = a / thetas.mean()
  else:
    b_hat = a_hat / thetas.mean()
  return a_hat, b_hat

def gamma_mle_correction(alpha, beta, N):
  corrected_a = alpha - 1/N *(3 * alpha - 2/3 * (alpha / (1 + alpha)) - 4/5 * (alpha / (1 + alpha)**2))
  corrected_b = (N - 1) / N * beta
  return corrected_a, corrected_b


from hyperbo.basics import lbfgs
from hyperbo.basics import bfgs
from hyperbo.basics import data_utils

# modified from GP class function
def infer_parameters_partial(mean_func,
                             cov_func,
                             init_params,
                             dataset,
                             warp_func=None,
                             objective=obj.neg_log_marginal_likelihood,
                             key=None,
                             get_params_path=None,
                             callback=None):
  """Posterior inference for a meta GP.
  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    init_params: GPParams, initial parameters for covariance, mean and noise
      variance, together with config parameters including 'method', a str
      indicating which method should be used. Currently it supports 'bfgs' and
      'momentum'.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    objective: objective loss function to minimize. Curently support
      neg_log_marginal_likelihood or sample_mean_cov_regularizer or linear
      combinations of them.
    key: Jax random state.
    get_params_path: optional function handle to return the path to save model
        params.
    callback: optional callback function for loggin of training steps.
  Returns:
    Dictionary of inferred parameters.
  """
  if not get_params_path:
    get_params_path = lambda x=0: None
  if key is None:
    key = jax.random.PRNGKey(0)
    logging.info('Using default random state in infer_parameters.')
  if not dataset:
    logging.info('No dataset present to train GP.')
    return init_params
  params = init_params
  method = params.config['method']
  batch_size = params.config['batch_size']
  # TODO(wangzi): clean this up.
  if method == 'lbfgs':
    # To handle very large sub datasets.
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
      subkey, dataset, batch_size)
    dataset = next(dataset_iter)

  max_training_step = init_params.config['max_training_step']

  if max_training_step <= 0 and method != 'slice_sample':
    return init_params

  if method == 'adam':
    @jit
    def loss_func(model_params, batch):
      return objective(
        mean_func=mean_func,
        cov_func=cov_func,
        params=GPParams(model=model_params, config=init_params.config),
        dataset=batch,
        warp_func=warp_func)

    optimizer = optax.adam(params.config['learning_rate'])
    opt_state = optimizer.init(params.model)

    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
      subkey, dataset, batch_size)
    model_param = params.model
    for i in range(max_training_step):
      batch = next(dataset_iter)
      current_loss, grads = jax.value_and_grad(loss_func)(model_param, batch)
      if jnp.isfinite(current_loss):
        params.model = model_param
      else:
        logging.info(msg=f'{method} stopped due to instability.')
        break
      updates, opt_state = optimizer.update(grads, opt_state)
      model_param = optax.apply_updates(model_param, updates)
      if callback:
        callback(i, params.model, current_loss)
    current_loss = loss_func(model_param, batch)
    if jnp.isfinite(current_loss):
      params.model = model_param
    params_utils.log_params_loss(
      step=max_training_step,
      params=params,
      loss=current_loss,
      warp_func=warp_func,
      params_save_file=get_params_path())
  else:
    def loss_func(model_params):
      return objective(
        mean_func=mean_func,
        cov_func=cov_func,
        params=GPParams(model=model_params, config=init_params.config),
        dataset=dataset,
        warp_func=warp_func)

    def loss_func_partial(lengthscale):
      model_params = params.model
      model_params['lengthscale'] = lengthscale
      return objective(
        mean_func=mean_func,
        cov_func=cov_func,
        params=GPParams(model=model_params, config=init_params.config),
        dataset=dataset,
        warp_func=warp_func)

    # TODO(nicole): add bfgs support
    if method == 'bfgs':
      params.model, _ = bfgs.bfgs(
        loss_func,
        params.model,
        tol=params.config['tol'],
        max_training_step=params.config['max_training_step'])
    elif method == 'lbfgs':

      if 'alpha' not in params.config:
        alpha = 1.0
      else:
        alpha = params.config['alpha']

      lengthscale = jnp.array(params.model['lengthscale'])
      current_loss, lengthscale, _ = lbfgs.lbfgs(
        loss_func_partial,
        lengthscale,
        steps=params.config['maxiter'],
        alpha=alpha,
        callback=callback)

      # params.model['lengthscale'] = utils.softplus_warp(lengthscale)
      params.model['lengthscale'] = lengthscale
      params_utils.log_params_loss(
        step=max_training_step,
        params=params,
        loss=current_loss,
        warp_func=warp_func,
        params_save_file=get_params_path())
    else:
      raise ValueError(f'Optimization method {method} is not supported.')
  params.cache = {}
  return params

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
import copy
import os

DEFAULT_WARP_FUNC = None
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params

kernel_list = [
  # ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
  ('matern32 nll', kernel.matern32, obj.nll, 'lbfgs'),
]

if __name__ == '__main__':
  n_workers = 1
  n_dim = 2
  n_discrete_points = 100
  n_test_funcs = 96
  budget = 50
  noise_variance = 1e-6
  length_scale = 0.05
  gp_fit_maxiter = 1
  different_domain = 1.0
  num_theta_samples = 2
  n_dataset_thetas = 50
  n_dataset_funcs = 3
  n_trials = 1

  alpha_gt = 1.
  beta_gt = 2.

  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count={}'.format(n_workers)

  pool = ProcessingPool(nodes=n_workers)

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
        'lengthscale': 1.0,  # this is just a placeholder, does not actually matter
        'signal_variance': 1.0,
        'noise_variance': noise_variance,
        'higher_params': [alpha_gt, beta_gt]
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
    data_list = hierarchical_gp.sample_from_gp(
      key, mean_func, cov_func, params, vx_dataset,
      n_dataset_thetas=n_dataset_thetas, n_dataset_funcs=n_dataset_funcs
    )
    dataset = []
    thetas = []
    for data in data_list:
      dataset_i = [(vx_dataset, data, 'all_data')]
      vy = dataset_i[0][1]
      for j in range(vy.shape[1]):
        dataset_i.append((vx_dataset, vy[:, j:j + 1]))
      dataset.append(dataset_i)

      dataset_k = [(vx_dataset, data, 'all_data')]
      vy = dataset_k[0][1]
      for j in range(vy.shape[1]):
        dataset_k.append((vx_dataset, vy[:, j:j + 1]))
      dataset.append(dataset_k)

      init_params_i = GPParams(
        model={
          'constant': 5.0,
          'lengthscale': 0.,
          'signal_variance': 1.,
          'noise_variance': 1e-6,
        },
        config={
          'method':
            opt_method,
          'maxiter':
            100,
          'max_training_step': 100,
          'logging_interval': 1,
          'objective': obj.nll,
          'batch_size': 100,
          'learning_rate': 0.001,
        })

      warp_func_i = utils.DEFAULT_WARP_FUNC_LC_ONLY

      model_i = gp.GP(
        dataset=dataset_i,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params_i,
        warp_func=warp_func_i)

      model_i.initialize_params(init_key)

      # inferred_params_i = model_i.train()
      inferred_params_i = infer_parameters_partial(
        mean_func=model_i.mean_func,
        cov_func=model_i.cov_func,
        init_params=init_params_i,
        dataset=model_i.dataset,
        warp_func=warp_func_i,
        objective=model_i.params.config['objective'])

      print("inferred: ", inferred_params_i)

      params_i = GPParams(
        model={
          'constant': 5.,
          'lengthscale': 0.5,
          'signal_variance': 1.0,
          'noise_variance': noise_variance,
        })
      keys_i = params_i.model.keys()
      print(keys_i)
      retrieved_inferred_params = dict(
        zip(keys_i, retrieve_params(inferred_params_i, keys_i, warp_func=warp_func_i)))
      print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))
      thetas.append(retrieved_inferred_params['lengthscale'])

    thetas = np.array(thetas)
    print("inferred lengthscales: ", thetas)
    print("alpha, beta if alpha is known: ", gamma_param_from_thetas(thetas, alpha_gt))
    print("alpha, beta if alpha is UNknown: ", gamma_param_from_thetas(thetas))

    #### Two-Level Inference

    # 1. Iterate through GPs dataset and find thetas

    # 2.A Use closed form MLE estimate of theta to find gamma params

    # TODO: 2.B Use LBFGS to find MLE parameters

  print('All done.')