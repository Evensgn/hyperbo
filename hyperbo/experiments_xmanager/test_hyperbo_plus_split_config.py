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
import subprocess
from experiment_defs import *


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit

if __name__ == '__main__':
    group_id = GROUP_ID
    results_dir = RESULTS_DIR
    normalize_x = True
    normalize_y = True

    is_hpob = True

    if is_hpob:
        train_id_list = HPOB_TRAIN_ID_LIST
        test_id_list = HPOB_TEST_ID_LIST
        setup_b_id_list = HPOB_FULL_ID_LIST

        hpob_negative_y = False
        hpob_data_path = HPOB_DATA_PATH
        dataset_func_combined = partial(data.hpob_dataset_v2, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                        normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hpob_dataset_v3, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                     normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_dim_feature_values_path = HPOB_DATA_ANALYSIS_PATH
        extra_info = 'hpob_negative_y: {}, noramlize_x: {}, normalize_y: {}'.format(hpob_negative_y, normalize_x, normalize_y)
    else:
        train_id_list = list(range(16))
        test_id_list = list(range(16, 20))
        setup_b_id_list = list(range(20))

        train_id_list = [str(x) for x in train_id_list]
        test_id_list = [str(x) for x in test_id_list]
        setup_b_id_list = [str(x) for x in setup_b_id_list]
        synthetic_data_path = SYNTHETIC_DATA_PTH
        dataset_func_combined = partial(data.hyperbo_plus_synthetic_dataset_combined, synthetic_data_path, normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hyperbo_plus_synthetic_dataset_split, synthetic_data_path, normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_dim_feature_values_path = SYNTHETIC_DATA_ANALYSIS_PATH
        extra_info = 'synthetic_data_path = \'{}\', noramlize_x: {}, normalize_y: {}'.format(synthetic_data_path, normalize_x, normalize_y)

    random_seed = RANDOM_SEED
    n_init_obs = 5
    budget = 50  # 50
    n_bo_runs = 5
    gp_fit_maxiter = 10000  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    n_bo_gamma_samples = 100  # 100
    n_nll_gamma_samples = 500  # 500
    setup_a_nll_sub_dataset_level = True
    fit_gp_batch_size = 50  # 50 for adam, 300 for lbfgs
    bo_sub_sample_batch_size = 1000  # 1000 for hpob, 300 for synthetic 4, 1000 for synthetic 5
    adam_learning_rate = 0.001
    eval_nll_batch_size = 100  # 300
    eval_nll_n_batches = 10
    ac_func_type_list = ['ucb', 'ei', 'pi']

    fixed_gp_distribution_params = {
        'constant': (0.0, 1.0),
        'lengthscale': (1.0, 10.0),
        'signal_variance': (1.0, 5.0),
        'noise_variance': (10.0, 100.0)
    }

    if is_hpob:
        gt_gp_distribution_params = None
    else:
        # ground truth for synthetic 4
        gt_gp_distribution_params = {
            'constant': (1.0, 1.0),
            'lengthscale': (10.0, 30.0),
            'signal_variance': (1.0, 1.0),
            'noise_variance': (10.0, 100000.0)
        }

    kernel_type = ('matern52 adam', kernel.matern52, obj.nll, obj.neg_log_marginal_likelihood_hgp_v3, 'adam')
    distribution_type = 'gamma'
    init_params_value_setup_b_path = INIT_PARAMS_VALUE_SETUP_B_PATH

    split_dir = os.path.join(results_dir, 'test_hyperbo_plus_split')
    dir_path = os.path.join(split_dir, group_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # write configs
    configs = {
        'random_seed': random_seed,
        'train_id_list': train_id_list,
        'test_id_list': test_id_list,
        'setup_b_id_list': setup_b_id_list,
        'dataset_func_combined': dataset_func_combined,
        'dataset_func_split': dataset_func_split,
        'dataset_dim_feature_values_path': dataset_dim_feature_values_path,
        'extra_info': extra_info,
        'n_init_obs': n_init_obs,
        'budget': budget,
        'n_bo_runs': n_bo_runs,
        'gp_fit_maxiter': gp_fit_maxiter,
        'n_bo_gamma_samples': n_bo_gamma_samples,
        'n_nll_gamma_samples': n_nll_gamma_samples,
        'setup_a_nll_sub_dataset_level': setup_a_nll_sub_dataset_level,
        'fit_gp_batch_size': fit_gp_batch_size,
        'bo_sub_sample_batch_size': bo_sub_sample_batch_size,
        'adam_learning_rate': adam_learning_rate,
        'eval_nll_batch_size': eval_nll_batch_size,
        'eval_nll_n_batches': eval_nll_n_batches,
        'ac_func_type_list': ac_func_type_list,
        'fixed_gp_distribution_params': fixed_gp_distribution_params,
        'gt_gp_distribution_params': gt_gp_distribution_params,
        'kernel_type': kernel_type,
        'distribution_type': distribution_type,
        'init_params_value_setup_b_path': init_params_value_setup_b_path,
    }
    np.save(os.path.join(dir_path, 'configs.npy'), configs)

