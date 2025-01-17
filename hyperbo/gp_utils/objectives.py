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

"""Objective functions for training a GP."""
import functools
import logging

from hyperbo.basics import linalg
from hyperbo.basics import params_utils
from hyperbo.gp_utils import utils
import jax.numpy as jnp
import jax
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma, LogNormal
from hyperbo.gp_utils import basis_functions as bf


retrieve_params = params_utils.retrieve_params


def sample_mean_cov_regularizer(mean_func,
                                cov_func,
                                params,
                                dataset,
                                warp_func=None,
                                distance=utils.kl_multivariate_normal,
                                use_feat0=False):
  """Compute a regularizer on sample mean and sample covariance.

  The returned regularizer aims to minimize the distance between the
  multivariate normal specified by sample mean/covariance and the multivariate
  normal specified by the parameterized GP. We support KL divergence as distance
  or squared Euclidean distance.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset. For aligned sub-dataset, this function should only be used if
      each aligned sub-dataset only has (?, m) for y shape, where m > 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    distance: distance function; currently support utils.kl_multivariate_normal
      or utils.euclidean_multivariate_normal.
    use_feat0: set to True if feat0 needs to be set in the distance function.

  Returns:
    Weighted l2 regularizer on sample mean and sample covariance.
  """

  def compute_regularizer_dataset_subset(sub_dataset):
    """Compute the regularizer on a subset of dataset keys."""
    if sub_dataset.y.shape[0] == 0:
      return 0.
    mu_data = jnp.mean(sub_dataset.y, axis=1)
    cov_data = jnp.cov(sub_dataset.y, bias=True)
    mu_model = mean_func(params, sub_dataset.x, warp_func=warp_func).flatten()
    noise_variance, = retrieve_params(
        params, ['noise_variance'], warp_func=warp_func)
    cov_model = cov_func(
        params, sub_dataset.x,
        warp_func=warp_func) + jnp.eye(sub_dataset.x.shape[0]) * noise_variance

    return distance(
        mu0=mu_data,
        cov0=cov_data,
        mu1=mu_model,
        cov1=cov_model,
        feat0=sub_dataset.y - mu_data[:, None] if use_feat0 else None)

  return jnp.sum(
      jnp.array([
          compute_regularizer_dataset_subset(sub_dataset)
          for sub_dataset in dataset.values()
          if sub_dataset.aligned is not None
      ]))


sample_mean_cov_regularizer_euc = functools.partial(
    sample_mean_cov_regularizer, distance=utils.euclidean_multivariate_normal)


def neg_log_marginal_likelihood(mean_func,
                                cov_func,
                                params,
                                dataset,
                                warp_func=None,
                                exclude_aligned=True):
  """Compute the negative of marginal likelihood of a (multi-task) GP.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    exclude_aligned: exclude sub-datasets that are aligned.

  Returns:
    Negative log marginal likelihood.
  """

  def compute_nll_sub_dataset(vx, vy):
    """Compute negative log likelihood for one sub dataset."""
    chol, kinvy, vy = linalg.solve_gp_linear_system(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x=vx,
        y=vy,
        warp_func=warp_func)
    nll_val = jnp.sum(0.5 * jnp.dot(vy.T, kinvy) +
                      jnp.sum(jnp.log(jnp.diag(chol))) +
                      0.5 * len(vx) * jnp.log(2 * jnp.pi))
    return nll_val

  total_nll = 0.
  for s in dataset.values():
    if exclude_aligned and s.aligned is not None:
      continue
    if s.x.shape[0] == 0:
      continue
    total_nll += compute_nll_sub_dataset(s.x, s.y)

  # We should really be including priors on the hyperparameters here.
  if 'priors' in params.config:
    for k in params.model:
      if k in params.config['priors']:
        log_prior_fn = params.config['priors'][k]
        val, = retrieve_params(params, [k], warp_func)
        log_prior_prob = log_prior_fn(val)
        logging.info(msg=f'log_prior_prob({k}={val}) = {log_prior_prob}')
        total_nll -= log_prior_prob
      else:
        logging.warning('No prior provided for param %s', k)

  return total_nll


def neg_log_marginal_likelihood_hgp_v2(key,
                                       mean_func,
                                       cov_func,
                                       params,
                                       super_dataset,
                                       dim_feature_values,
                                       warp_func=None,
                                       exclude_aligned=True,
                                       n_gamma_samples=1):
  """Compute the negative of marginal likelihood of a (multi-task) GP.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    exclude_aligned: exclude sub-datasets that are aligned.

  Returns:
    Negative log marginal likelihood.
  """

  def compute_nll_sub_dataset(key, vx, vy, dim_feature_value):
    """Compute negative log likelihood for one sub dataset."""
    chol, kinvy, vy = linalg.solve_gp_linear_system_hgp_v2(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        dim_feature_value=dim_feature_value,
        x=vx,
        y=vy,
        warp_func=warp_func)
    nll_val = jnp.sum(0.5 * jnp.dot(vy.T, kinvy) +
                      jnp.sum(jnp.log(jnp.diag(chol))) +
                      0.5 * len(vx) * jnp.log(2 * jnp.pi))
    return nll_val

  total_nll = 0.
  for dataset_key, dataset in super_dataset.items():
      print('dataset_key:', dataset_key)
      dim_feature_value = dim_feature_values[dataset_key]
      for sub_dataset_key, sub_dataset in dataset.items():
        print('sub_dataset_key:', sub_dataset_key)
        if exclude_aligned and sub_dataset.aligned is not None:
          continue
        if sub_dataset.x.shape[0] == 0:
          continue
        for _ in range(n_gamma_samples):
            key, sub_key = jax.random.split(key)
            total_nll += compute_nll_sub_dataset(sub_key, sub_dataset.x, sub_dataset.y, dim_feature_value)
  total_nll /= n_gamma_samples

  return total_nll


def neg_log_marginal_likelihood_hgp_v3(mean_func,
                                       cov_func,
                                       params,
                                       super_dataset,
                                       dim_feature_values,
                                       hgp_warp_func=None,
                                       single_gp_warp_func=None,
                                       exclude_aligned=True):
  """Compute the negative of marginal likelihood of a (multi-task) GP.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    exclude_aligned: exclude sub-datasets that are aligned.

  Returns:
    Negative log marginal likelihood.
  """

  def compute_nll_sub_dataset(dataset_key, vx, vy):
    """Compute negative log likelihood for one sub dataset."""
    chol, kinvy, vy = linalg.solve_gp_linear_system_hpg_v3(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        dataset_key=dataset_key,
        x=vx,
        y=vy,
        single_gp_warp_func=single_gp_warp_func,
    )
    nll_val = jnp.sum(0.5 * jnp.dot(vy.T, kinvy) +
                      jnp.sum(jnp.log(jnp.diag(chol))) +
                      0.5 * len(vx) * jnp.log(2 * jnp.pi))
    return nll_val

  def compute_nll_distribution_params():
      nll = 0.
      distribution_type = params.config['distribution_type']
      warpped_model = utils.apply_warp_func(params.model, hgp_warp_func)

      constant_mu, constant_sigma = warpped_model['constant_normal_params']
      constant_dist = Normal(constant_mu, constant_sigma)

      if distribution_type == 'gamma':
          signal_variance_a, signal_variance_b = warpped_model['signal_variance_gamma_params']
          signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
          noise_variance_a, noise_variance_b = warpped_model['noise_variance_gamma_params']
          noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
      elif distribution_type == 'lognormal':
          signal_variance_mu, signal_variance_sigma = warpped_model['signal_variance_lognormal_params']
          signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
          noise_variance_mu, noise_variance_sigma = warpped_model['noise_variance_lognormal_params']
          noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
      else:
          raise ValueError('Unknown distribution type: {}'.format(distribution_type))

      for dataset_key in super_dataset.keys():
          dim_feature_value = dim_feature_values[dataset_key]
          params_value = warpped_model['search_space_params'][dataset_key]
          constant = params_value['constant']
          nll += -constant_dist.log_prob(constant)
          signal_variance = params_value['signal_variance']
          nll += -signal_variance_dist.log_prob(signal_variance)
          noise_variance = params_value['noise_variance']
          nll += -noise_variance_dist.log_prob(noise_variance)
          lengthscale = params_value['lengthscale']
          if distribution_type == 'gamma':
              lengthscale_dist_mlp_features = params.config['lengthscale_gamma_mlp_features']
              lengthscale_dist_mlp_params = params.model['lengthscale_gamma_mlp_params']
          elif distribution_type == 'lognormal':
              lengthscale_dist_mlp_features = params.config['lengthscale_lognormal_mlp_features']
              lengthscale_dist_mlp_params = params.model['lengthscale_lognormal_mlp_params']
          else:
              raise ValueError('Unknown distribution type: {}'.format(distribution_type))
          lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
          lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params}, dim_feature_value)
          for dim in range(len(lengthscale)):
              if distribution_type == 'gamma':
                  lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
                  lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                  lengthscale_dist = Gamma(lengthscale_a, rate=lengthscale_b)
              elif distribution_type == 'lognormal':
                  lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
                  lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                  lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
              else:
                  raise ValueError('Unknown distribution type: {}'.format(distribution_type))
              nll += -lengthscale_dist.log_prob(lengthscale[dim])
      return nll


  total_nll = 0.
  for dataset_key, dataset in super_dataset.items():
      for sub_dataset_key, sub_dataset in dataset.items():
        if exclude_aligned and sub_dataset.aligned is not None:
          continue
        if sub_dataset.x.shape[0] == 0:
          continue
        total_nll += compute_nll_sub_dataset(dataset_key, sub_dataset.x, sub_dataset.y)

  total_nll += compute_nll_distribution_params()

  return total_nll


def neg_log_marginal_likelihood_tuple(args):
    return neg_log_marginal_likelihood(*args)


nll_tuple = neg_log_marginal_likelihood_tuple

nll = neg_log_marginal_likelihood
regkl = sample_mean_cov_regularizer
regeuc = sample_mean_cov_regularizer_euc
kl = regkl


def add(*objectives):

  def added_objective(*args, **kwargs):
    return sum([obj(*args, **kwargs) for obj in objectives])

  return added_objective


def mul(c, obj):

  def multiplied_objective(*args, **kwargs):
    return c * obj(*args, **kwargs)

  return multiplied_objective


nll_regkl = lambda c: add(nll, mul(c, regkl))
nll_regeuc = lambda c: add(nll, mul(c, regeuc))

nll_regkl1 = nll_regkl(1.)
nll_regeuc1 = nll_regeuc(1.)
nll_regkl01 = nll_regkl(.1)
nll_regeuc01 = nll_regkl(.1)

nll_regkl10 = nll_regkl(10.)
nll_regeuc10 = nll_regkl(10.)
