# This file is based on gp.py from HyperBO
# original copyright and license info below:

# coding=utf-8
# Copyright 2022 HyperBO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Inferrence and other util functions for a hierarchical GP."""
# only generates the scalar length_scale parameter from a Gamma distribution at this point


import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flax
from flax import linen as nn
from hyperbo.basics import bfgs
from hyperbo.basics import data_utils
from hyperbo.basics import definitions as defs
from hyperbo.basics import lbfgs
from hyperbo.basics import linalg
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
import jax
from jax import flatten_util
import jax.numpy as jnp
import jax.random
import jax.scipy as jsp
import optax
import numpy as np
from tensorflow_probability.substrates.jax.distributions import Gamma
import copy


grad = jax.grad
jit = jax.jit
vmap = jax.vmap

retrieve_params = params_utils.retrieve_params

GPCache = defs.GPCache
SubDataset = defs.SubDataset
GPParams = defs.GPParams


def infer_parameters(mean_func,
                     cov_func,
                     init_params,
                     dataset,
                     warp_func=None,
                     objective=obj.neg_log_marginal_likelihood,
                     key=None,
                     params_save_file=None,
                     num_theta_samples=100,
                     n_workers=1):
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
    params_save_file: optional file name to save params.

  Returns:
    Dictionary of inferred parameters.
  """
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
    pass
    '''
    # To handle very large sub datasets.
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        subkey, dataset, batch_size)
    dataset = next(dataset_iter)
    '''

  maxiter = init_params.config['maxiter']
  logging_interval = init_params.config['logging_interval']

  if maxiter <= 0 and method != 'slice_sample':
    return init_params

  # adam not supported
  if method == 'adam':
    raise ValueError('adam not supported')
  else:
    # @jit
    def loss_func(higher_params):
      nonlocal key
      nonlocal dataset
      model_params = params.model

      # sample gp params first
      higher_params = utils.softplus_warp(higher_params)
      # gamma_alpha, gamma_beta = higher_params[0], higher_params[1]
      gamma_alpha, gamma_beta = 0.5, higher_params[0]
      gamma = Gamma(gamma_alpha, gamma_beta)
      key, _ = jax.random.split(key, 2)
      thetas = gamma.sample(num_theta_samples, seed=key)

      def loss_dataset_theta(dataset, theta):
          model_params['lengthscale'] = theta
          return objective(
              mean_func,
              cov_func,
              copy.deepcopy(GPParams(model=model_params, config=init_params.config)),
              dataset,
              warp_func,
              True
          )

      loss = 0.0
      for dataset_i in dataset:
          loss_theta = lambda theta: loss_dataset_theta(dataset_i, theta)
          B = (num_theta_samples - 1) // n_workers + 1
          objectives = []
          for b in range(B):
              objectives.append(jax.pmap(loss_theta)(thetas[b * n_workers:min((b + 1) * n_workers, num_theta_samples)]))
          objectives = jnp.concatenate(objectives, axis=None)
          loss_i = -jax.scipy.special.logsumexp(-objectives, axis=0) - jnp.log(num_theta_samples)
          # print('loss_i:', loss_i)
          loss += loss_i

      return loss

    if method == 'bfgs':
      higher_params = jnp.array([2.0]) # hard coded for now
      higher_params, _ = bfgs.bfgs(
          loss_func,
          higher_params,
          tol=params.config['tol'],
          maxiter=params.config['maxiter'])
      params.model['higher_params'] = utils.softplus_warp(higher_params)
    elif method == 'lbfgs':
      def lbfgs_callback(step, model_params, loss):
        if step % logging_interval != 0:
          return
        params.model = model_params
        params_utils.log_params_loss(
            step,
            params,
            loss,
            warp_func=warp_func,
            params_save_file=params_save_file)

      if 'alpha' not in params.config:
        alpha = 1.0
      else:
        alpha = params.config['alpha']
      higher_params = jnp.array([2.0]) # hard coded for now
      current_loss, higher_params, _ = lbfgs.lbfgs(
          loss_func,
          higher_params,
          steps=params.config['maxiter'],
          alpha=alpha,
          callback=None)
      params.model['higher_params'] = utils.softplus_warp(higher_params)

      '''
      params_utils.log_params_loss(
          step=maxiter,
          params=params,
          loss=current_loss,
          warp_func=warp_func,
          params_save_file=params_save_file)
      '''
    else:
      raise ValueError(f'Optimization method {method} is not supported.')
  params.cache = {}
  return params


def sample_from_gp(key,
                   mean_func,
                   cov_func,
                   params,
                   x,
                   warp_func=None,
                   n_dataset_thetas=10,
                   n_dataset_funcs=10,
                   method='cholesky',
                   eps=1e-6):
  """Sample a function from a GP and return its evaluations on x (n x d).

  Args:
    key: a pseudo-random number generator (PRNG) key for jax.random.
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    x: n x d dimensional input array.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    num_samples: number of draws from the multivariate normal.
    method: the decomposition method to invert the cov matrix.
    eps: extra value added to the diagonal of the covariance matrix for
      numerical stability.

  Returns:
    n x num_samples funcion evaluations on x. Each function is a sample from the
    GP.
  """
  # sample gp params first
  higher_params = params.model['higher_params']
  # gamma_alpha, gamma_beta = higher_params[0], higher_params[1]
  gamma_alpha, gamma_beta = 0.5, higher_params[0]
  gamma = Gamma(gamma_alpha, gamma_beta)
  key, _ = jax.random.split(key, 2)
  thetas = gamma.sample(n_dataset_thetas, seed=key)
  # sample functions for each theta and concatenate
  datasets = []
  for theta in thetas:
    params.model['lengthscale'] = theta
    mean = mean_func(params, x, warp_func=warp_func)
    noise_variance, = retrieve_params(
        params, ['noise_variance'], warp_func=warp_func)
    cov = cov_func(params, x, warp_func=warp_func)
    key, _ = jax.random.split(key, 2)
    theta_dataset = (jax.random.multivariate_normal(
      key,
      mean.flatten(),
      cov + jnp.eye(len(x)) * (noise_variance + eps),
      shape=(n_dataset_funcs,),
      method=method)).T
    datasets.append(theta_dataset)
  return datasets


# predict does not work yet
def predict(mean_func,
            cov_func,
            params,
            x_observed,
            y_observed,
            x_query,
            warp_func=None,
            full_cov=False,
            cache=None):
  """Predict the gp for query points x_query conditioned on observations.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    x_observed: observed n x d input array.
    y_observed: observed n x 1 evaluations on the input x_observed.
    x_query: n' x d input array to be queried.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    full_cov: flag for returning full covariance if true or variance if false.
    cache: GPCache, cached Cholesky decomposition of gram matrix K on x_observed
      and cached K inverse times y_observed (mean subtracted).

  Returns:
    Predicted posterior mean (n' x 1) and covariance (n' x n') (or n' x 1
    variance terms if full_cov=False) for query points
    x_query.
  """
  raise NotImplementedError('predict not implemented yet')

  if x_observed is None or x_observed.shape[0] == 0:
    # Compute the prior distribution.
    mu = mean_func(params, x_query, warp_func=warp_func)
    cov = cov_func(params, x_query, warp_func=warp_func, diag=not full_cov)
    if full_cov:
      return mu, cov
    else:
      return mu, cov[:, None]

  # One can potentially support rank-1 updates.
  if cache is None:
    chol, kinvy, _ = linalg.solve_gp_linear_system(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x=x_observed,
        y=y_observed,
        warp_func=warp_func)
  else:
    chol, kinvy = cache.chol, cache.kinvy
  cov = cov_func(params, x_observed, x_query, warp_func=warp_func)
  mu = jnp.dot(cov.T, kinvy) + mean_func(params, x_query, warp_func=warp_func)
  v = jsp.linalg.solve_triangular(chol, cov, lower=True)
  if full_cov:
    cov = cov_func(params, x_query, warp_func=warp_func) - jnp.dot(v.T, v)
    return mu, cov
  else:
    diagdot = vmap(lambda x: jnp.dot(x, x.T))
    var = cov_func(
        params, x_query, warp_func=warp_func, diag=True) - diagdot(v.T)
    return mu, var[:, None]


class HierarchicalGP:
  """A Gaussian process that supports learning with historical data.

  Attributes:
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: GPParams, parameters for covariance, mean, and noise variance.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    input_dim: dimension of input variables.
    rng: Jax random state.
  """
  dataset: List[Dict[Union[int, str], SubDataset]]

  def __init__(self,
               dataset: List[Union[List[Union[Tuple[jnp.ndarray, ...], SubDataset]],
                              Dict[Union[str], Union[Tuple[jnp.ndarray, ...],
                                                     SubDataset]]]],
               mean_func: Callable[..., jnp.array],
               cov_func: Callable[..., jnp.array],
               params: GPParams,
               warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None,
               pool=None):
    self.mean_func = mean_func
    self.cov_func = cov_func
    if params is not None:
      self.params = params
    else:
      self.params = GPParams()
    self.warp_func = warp_func
    self.set_dataset(dataset)
    if 'objective' not in self.params.config:
      self.params.config['objective'] = obj.neg_log_marginal_likelihood
    self.rng = None
    self.pool = pool

  def initialize_params(self, key):
    """Initialize params with a JAX random state."""
    if not self.dataset:
      raise ValueError('Cannot initialize GPParams without dataset.')
    logging.info(msg=f'dataset: {jax.tree_map(jnp.shape, self.dataset)}')

    if isinstance(self.params.config['objective'], str):
      self.params.config['objective'] = getattr(obj,
                                                self.params.config['objective'])
    def check_param(name, param_type, params_dict=self.params.model):
      return name in params_dict and isinstance(
          params_dict[name], param_type)

    if 'mlp' in self.mean_func.__name__ or 'mlp' in self.cov_func.__name__:
      if not check_param('mlp_features', tuple, self.params.config):
        self.params.config['mlp_features'] = (2 * self.input_dim,)
      last_layer_size = self.params.config['mlp_features'][-1]
      if check_param('mlp_params', flax.core.frozen_dict.FrozenDict):
        flag = 'Retained'
      else:
        key, subkey = jax.random.split(key)
        bf.init_mlp_with_shape(subkey, self.params, (0, self.input_dim))
        flag = 'Initialized'
      mlp_params = self.params.model['mlp_params']
      logging.info(msg=f'{flag} mlp_params: '
                   f'{jax.tree_map(jnp.shape, mlp_params)}')
    else:
      last_layer_size = self.input_dim
    if 'linear' in self.mean_func.__name__:
      if check_param('linear_mean', flax.core.frozen_dict.FrozenDict):
        flag = 'Retained'
      else:
        key, subkey = jax.random.split(key)
        self.params.model['linear_mean'] = nn.Dense(1).init(
            subkey, jnp.empty((0, last_layer_size)))['params']
        flag = 'Initialized'

      linear_mean = self.params.model['linear_mean']
      logging.info(msg=f'{flag} linear_mean: '
                   f'{jax.tree_map(jnp.shape, linear_mean)}')
    if self.cov_func in [
        kernel.dot_product_kumar, kernel.dot_product_mlp, kernel.dot_product
    ]:
      if check_param('dot_prod_sigma', jnp.ndarray) and check_param(
          'dot_prod_bias', float):
        flag = 'Retained'
      else:
        key, subkey = jax.random.split(key)
        self.params.model['dot_prod_sigma'] = jax.random.normal(
            subkey, (last_layer_size, last_layer_size * 2))
        key, subkey = jax.random.split(key)
        self.params.model['dot_prod_bias'] = jax.random.normal(subkey)
        flag = 'Initialized'
      dot_prod_sigma = self.params.model['dot_prod_sigma']
      dot_prod_bias = self.params.model['dot_prod_bias']
      logging.info(msg=f'{flag} dot_prod_sigma: '
                   f'{dot_prod_sigma.shape} and dot_prod_bias: {dot_prod_bias}')
    self.rng = key

  def set_dataset(self, dataset: List[Union[List[Union[Tuple[jnp.ndarray, ...],
                                                  SubDataset]],
                                       Dict[Union[int, str],
                                            Union[Tuple[jnp.ndarray, ...],
                                                  SubDataset]]]]):
    """Reset GP dataset to be dataset.

    Args:
      dataset: a list of vx, vy pairs, i.e. [(vx, vy)_i], where vx is n x d and
        vy is n x 1.
    """
    self.dataset = []
    for i, dataset_i in enumerate(dataset):
        new_dataset_i = {}
        if isinstance(dataset_i, list):
          dataset_i = {j: dataset_i[j] for j in range(len(dataset_i))}
        for key, val in dataset_i.items():
          new_dataset_i[key] = SubDataset(*val)
        self.dataset.append(new_dataset_i)

  @property
  def input_dim(self) -> int:
    key = list(self.dataset[0].keys())[0]
    return self.dataset[0][key].x.shape[1]

  def update_sub_dataset(self,
                         sub_dataset: Union[Tuple[jnp.ndarray, ...],
                                            SubDataset],
                         sub_dataset_key: Union[int, str] = 0,
                         is_append: bool = False):
    """Update a sub-dataset in dataset.

    Args:
      sub_dataset: the new sub-dataset as in (vx, vy) pair to be updated to
        dataset.
      sub_dataset_key: the key of the sub-dataset in dataset.
      is_append: append to the sub-dataset if True; otherwise replace it.
    """
    sub_dataset = SubDataset(*sub_dataset)

    if is_append:
      if sub_dataset_key not in self.dataset:
        assert self.dataset, 'dataset cannot be empty.'
        self.dataset[sub_dataset_key] = SubDataset(
            x=jnp.empty((0, self.input_dim)), y=jnp.empty((0, 1)))
      new_x = jnp.vstack((self.dataset[sub_dataset_key].x, sub_dataset.x))
      new_y = jnp.vstack((self.dataset[sub_dataset_key].y, sub_dataset.y))
      self.dataset[sub_dataset_key] = SubDataset(x=new_x, y=new_y)
    else:
      self.dataset[sub_dataset_key] = sub_dataset
    if sub_dataset_key in self.params.cache:
      self.params.cache[sub_dataset_key].needs_update = True

  def train(self, key=None, params_save_file=None, n_workers=1) -> GPParams:
    """Train the GP by fitting it to the dataset.

    Args:
      key: Jax random state.
      params_save_file: optional file name to save params.

    Returns:
      params: GPParams.
    """
    if key is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
        logging.info('Using default random state in GP.train.')
      key, subkey = jax.random.split(self.rng, 2)
      self.rng = key
    else:
      key, subkey = jax.random.split(key, 2)
    self.params = infer_parameters(
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        init_params=self.params,
        dataset=self.dataset,
        warp_func=self.warp_func,
        objective=self.params.config['objective'],
        key=subkey,
        params_save_file=params_save_file,
        num_theta_samples=self.params.config['num_theta_samples'],
        n_workers=n_workers
    )
    logging.info(msg=f'params = {self.params}')
    return self.params

  # does not work for now
  def neg_log_marginal_likelihood(self) -> float:
    """Compute negative log marginal likelihood for current model."""
    return obj.neg_log_marginal_likelihood(
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        params=self.params,
        dataset=self.dataset,
        warp_func=self.warp_func)

  # does not work for now
  def sample_mean_cov_regularizer(self,
                                  distance=utils.kl_multivariate_normal
                                 ) -> float:
    """Compute regularizer on sample mean and sample covariance."""
    return obj.sample_mean_cov_regularizer(
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        params=self.params,
        dataset=self.dataset,
        warp_func=self.warp_func,
        distance=distance,
        use_feat0=True)

  # does not work for now
  def stats(self, verbose=True) -> Tuple[float, float, float]:
    """Compute objective stats for current model."""
    nll = self.neg_log_marginal_likelihood()
    klreg = self.sample_mean_cov_regularizer(
        distance=functools.partial(
            utils.kl_multivariate_normal, partial=False, eps=1e-6))
    eucreg = self.sample_mean_cov_regularizer(
        distance=utils.euclidean_multivariate_normal)
    msg = f'nll = {nll}, kl reg = {klreg}, euc reg = {eucreg}'
    if verbose:
      print(msg)
    logging.info(msg=msg)
    return nll, klreg, eucreg

  def update_model_params(self, model_params: Dict[str, Any]):
    """Update params.model (must clean up params.cache)."""
    self.params.model = model_params
    self.params.cache = {}

  def setup_predictor(self, sub_dataset_key: Union[int, str] = 0):
    """Set up the GP predictor by computing its cached parameters.

    Args:
      sub_dataset_key: the key of the sub-dataset to run predict.

    Returns:
      params: GPParams.
    """
    if sub_dataset_key in self.params.cache and not self.params.cache[
        sub_dataset_key].needs_update:
      return
    chol, kinvy, _ = linalg.solve_gp_linear_system(
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        params=self.params,
        x=self.dataset[sub_dataset_key].x,
        y=self.dataset[sub_dataset_key].y,
        warp_func=self.warp_func)
    self.params.cache[sub_dataset_key] = GPCache(
        chol=chol, kinvy=kinvy, needs_update=False)

  # predict does not work yet
  def predict(self,
              queried_inputs: jnp.ndarray,
              sub_dataset_key: Union[int, str] = 0,
              full_cov: bool = False,
              with_noise: bool = True,
              unbiased: bool = True) -> Tuple[jnp.array, jnp.array]:
    """Predict the distribution of evaluations on queried_inputs.

    Args:
      queried_inputs: queried inputs (n' x d).
      sub_dataset_key: the key of the sub-dataset to run predict.
      full_cov: Flag for returning full covariance if true or variance if false.
      with_noise: Flag for returning (co)variance with observation noize if True
        or without observation noize if False.
      unbiased: if True, rescale predicted covariance by N/(N-1).

    Returns:
      mu: predicted n' x 1 mean.
      cov: predicted n' x n' covariance if full_cov; otherwise n' x 1 variance.
    """
    raise NotImplementedError('predict not implemented yet')

    if sub_dataset_key not in self.dataset:
      mu, cov = predict(
          mean_func=self.mean_func,
          cov_func=self.cov_func,
          params=self.params,
          x_observed=None,
          y_observed=None,
          x_query=queried_inputs,
          warp_func=self.warp_func,
          full_cov=full_cov)
    else:
      self.setup_predictor(sub_dataset_key)
      mu, cov = predict(
          mean_func=self.mean_func,
          cov_func=self.cov_func,
          params=self.params,
          x_observed=self.dataset[sub_dataset_key].x,
          y_observed=self.dataset[sub_dataset_key].y,
          x_query=queried_inputs,
          warp_func=self.warp_func,
          full_cov=full_cov,
          cache=self.params.cache[sub_dataset_key])

    if with_noise:
      noise_variance, = retrieve_params(
          self.params, ['noise_variance'], warp_func=self.warp_func)
      if full_cov:
        cov += jnp.eye(cov.shape[0]) * (noise_variance)
      else:
        cov += noise_variance
    if unbiased:
      len_dataset = len(
          [k for k, v in self.dataset.items() if v.aligned is None])
      if len_dataset > 1:
        scale = len_dataset / (len_dataset - 1.)
        cov *= scale
    return mu, cov