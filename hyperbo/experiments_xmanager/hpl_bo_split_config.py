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

    is_hpob = IS_HPOB

    train_id_list = TRAIN_ID_LIST
    test_id_list = TEST_ID_LIST
    setup_b_id_list = FULL_ID_LIST

    if is_hpob:
        hpob_negative_y = False
        hpob_data_path = HPOB_DATA_PATH
        dataset_func_combined = partial(data.hpob_dataset_v2, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                        normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hpob_dataset_v3, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                     normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_dim_feature_values_path = HPOB_DATA_ANALYSIS_PATH
        extra_info = 'hpob_negative_y: {}, noramlize_x: {}, normalize_y: {}'.format(hpob_negative_y, normalize_x, normalize_y)
    else:
        train_id_list = [str(x) for x in train_id_list]
        test_id_list = [str(x) for x in test_id_list]
        setup_b_id_list = [str(x) for x in setup_b_id_list]
        synthetic_data_path = SYNTHETIC_DATA_PTH
        dataset_func_combined = partial(data.hyperbo_plus_synthetic_dataset_combined, synthetic_data_path,
                                        normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hyperbo_plus_synthetic_dataset_split, synthetic_data_path,
                                     normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_dim_feature_values_path = SYNTHETIC_DATA_ANALYSIS_PATH
        extra_info = 'synthetic_data_path = \'{}\', noramlize_x: {}, normalize_y: {}'.format(synthetic_data_path, normalize_x, normalize_y)

    random_seed = RANDOM_SEED

    fit_gp_maxiter = 10000  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    fit_gp_batch_size = 50  # 50 for adam, 300 for lbfgs
    fit_gp_adam_learning_rate = 0.001

    fit_hgp_maxiter = 10000
    fit_hgp_batch_size = 50
    fit_hgp_adam_learning_rate = 0.001

    fit_two_step_maxiter = 5000
    fit_two_step_learning_rate = 0.01

    n_init_obs = 5
    budget = 100  # 50
    n_bo_runs = 5
    n_bo_gp_params_samples = 100  # 100
    n_nll_gp_params_samples = 500  # 500
    bo_sub_sample_batch_size = None  # 1000 for hpob, 300 for synthetic 4, 1000 for synthetic 5

    eval_nll_batch_size = 100  # 300
    eval_nll_n_batches = 10
    ac_func_type_list = ['ucb', 'ei', 'pi']

    hand_hgp_params = {
        'constant': (0.0, 1.0),
        'lengthscale': (1.0, 10.0),
        'signal_variance': (1.0, 5.0),
        'noise_variance': (10.0, 100.0)
    }

    uniform_hgp_params = {
        'constant': (-100, 100.0),
        'lengthscale': (0.001, 10.0),
        'signal_variance': (0.000001, 100.0),
        'noise_variance': (0.00000001, 100.0)
    }

    method_name_list = ['random', 'hyperbo', 'hand_hgp', 'uniform_hgp', 'fit_direct_hgp', 'fit_direct_hgp_leaveout',
                        'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_leaveout', 'hpl_hgp_end_to_end_from_scratch',
                        'hpl_hgp_end_to_end_leaveout_from_scratch', 'hpl_hgp_two_step', 'hpl_hgp_two_step_leaveout']
    setup_b_only_method_name_list = ['hyperbo', 'fit_direct_hgp_leaveout', 'hpl_hgp_end_to_end_leaveout',
                                     'hpl_hgp_end_to_end_leaveout_from_scratch', 'hpl_hgp_two_step_leaveout']

    if is_hpob:
        gt_hgp_params = None
    else:
        # ground truth for synthetic 4
        gt_hgp_params = {
            'constant': (1.0, 1.0),
            'lengthscale': (10.0, 30.0),
            'signal_variance': (1.0, 1.0),
            'noise_variance': (10.0, 100000.0)
        }
        method_name_list.append('gt_hgp')

    kernel_type = ('matern52 adam', kernel.matern52, obj.nll, obj.neg_log_marginal_likelihood_hgp_v3, 'adam')
    mean_func = mean.constant
    distribution_type = 'gamma'
    fitting_node_cpu_count = FITTING_NODE_CPU_COUNT
    bo_node_cpu_count = BO_NODE_CPU_COUNT
    nll_node_cpu_count = NLL_NODE_CPU_COUNT

    split_dir = os.path.join(results_dir, 'hpl_bo_split')
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

        'fit_gp_maxiter': fit_gp_maxiter,
        'fit_gp_batch_size': fit_gp_batch_size,
        'fit_gp_adam_learning_rate': fit_gp_adam_learning_rate,

        'fit_hgp_maxiter': fit_hgp_maxiter,
        'fit_hgp_batch_size': fit_hgp_batch_size,
        'fit_hgp_adam_learning_rate': fit_hgp_adam_learning_rate,

        'fit_two_step_maxiter': fit_two_step_maxiter,
        'fit_two_step_learning_rate': fit_two_step_learning_rate,

        'n_init_obs': n_init_obs,
        'budget': budget,
        'n_bo_runs': n_bo_runs,
        'n_bo_gp_params_samples': n_bo_gp_params_samples,
        'n_nll_gp_params_samples': n_nll_gp_params_samples,
        'bo_sub_sample_batch_size': bo_sub_sample_batch_size,
        'eval_nll_batch_size': eval_nll_batch_size,
        'eval_nll_n_batches': eval_nll_n_batches,
        'ac_func_type_list': ac_func_type_list,

        'hand_hgp_params': hand_hgp_params,
        'uniform_hgp_params': uniform_hgp_params,
        'gt_hgp_params': gt_hgp_params,

        'kernel_type': kernel_type,
        'mean_func': mean_func,
        'distribution_type': distribution_type,

        'fitting_node_cpu_count': fitting_node_cpu_count,
        'bo_node_cpu_count': bo_node_cpu_count,
        'nll_node_cpu_count': nll_node_cpu_count,

        'method_name_list': method_name_list,
        'setup_b_only_method_name_list': setup_b_only_method_name_list,
    }
    np.save(os.path.join(dir_path, 'configs.npy'), configs)

