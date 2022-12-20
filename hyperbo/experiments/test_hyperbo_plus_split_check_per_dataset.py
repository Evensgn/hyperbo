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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


hpob_full_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                     '6794', '7607', '7609', '5889']

kernel_list = [
    # ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
    # ('matern32 adam', kernel.matern32, obj.nll, 'adam'),
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
    # ('matern52 kl', kerne.matern52, obj.kl, 'lbfgs'),
    # ('matern32_mlp kl', kernel.matern32_mlp, obj.kl, 'lbfgs'),
    # ('matern52_mlp kl', kernel.matern52_mlp, obj.kl, 'lbfgs'),
    # ('squared_exponential_mlp kl', kernel.squared_exponential_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl', kernel.dot_product_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl adam', kernel.dot_product_mlp, obj.kl, 'adam'),
    # ('squared_exponential_mlp kl adam', kernel.squared_exponential_mlp, obj.kl, 'adam')
]

param_names = {
    'constant': 'Constant',
    'lengthscale': 'Length-scale',
    'signal_variance': 'Signal Variance',
    'noise_variance': 'Noise Variance',
}


if __name__ == '__main__':
    n_train_datasets_list = list(range(2, 17))
    n_seeds = 5
    # aggregation_dir_path = os.path.join('results', 'test_hyperbo_plus_split_hpob_pos_1_50_check_per_dataset')
    aggregation_dir_path = os.path.join('results', 'test_hyperbo_plus_split_synthetic_2_50_check_per_dataset')

    # separation
    '''
    # train_id_list = ['5860', '5906']
    # test_id_list = ['5889']
    # train_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
    # train_id_list = ['6766', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6767', '6794']
    # train_id_list = ['4796', '5527', '5636', '5859', '5891', '5965', '5970', '5971', '6766', '6767', '6794', '7607', '7609']
    # train_id_list = ['4796', '5527', '5636']
    # test_id_list = ['4796', '5860', '5906', '7607', '7609', '5889']
    # test_id_list = ['5860', '5906', '5889']
    # setup_b_id_list = ['4796', '5860', '5906', '7607', '7609', '5889']
    # setup_b_id_list = ['5860', '5906', '5889']

    # train_id_list = ['4796', '5527']
    train_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
    test_id_list = ['6794', '7607', '7609', '5889']
    setup_b_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                       '6794', '7607', '7609', '5889']

    # train_id_list = ['4796', '5527', '5636', '5859', '5860']
    # test_id_list = ['6794', '7607']
    # setup_b_id_list = ['4796', '5527', '5636', '5859', '5860']

    hpob_negative_y = False
    dataset_func_combined = partial(data.hpob_dataset_v2, negative_y=hpob_negative_y)
    dataset_func_split = partial(data.hpob_dataset_v3, negative_y=hpob_negative_y)
    extra_info = 'hpob_negative_y_{}'.format(hpob_negative_y)

    # hpob_converted_data_path = './hpob_converted_data/sub_sample_1000.npy'
    # dataset_func_combined = partial(data.hpob_converted_dataset_combined, hpob_converted_data_path)
    # dataset_func_split = partial(data.hpob_converted_dataset_split, hpob_converted_data_path)
    # extra_info = 'hpob_converted_data_path = \'{}\''.format(hpob_converted_data_path)
    '''

    train_id_list = list(range(16))
    test_id_list = list(range(16, 20))
    setup_b_id_list = list(range(20))

    test_id_list = [str(x) for x in test_id_list]
    setup_b_id_list = [str(x) for x in setup_b_id_list]
    synthetic_data_path = './synthetic_data/dataset_4.npy'
    dataset_func_combined = partial(data.hyperbo_plus_synthetic_dataset_combined, synthetic_data_path)
    dataset_func_split = partial(data.hyperbo_plus_synthetic_dataset_split, synthetic_data_path)
    extra_info = 'synthetic_data_path = \'{}\''.format(synthetic_data_path)

    n_workers = 25
    n_init_obs = 5
    budget = 50  # 50
    n_bo_runs = 5
    gp_fit_maxiter = 500  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    n_bo_gamma_samples = 100  # 100
    n_nll_gamma_samples = 500  # 500
    setup_a_nll_sub_dataset_level = True
    fit_gp_batch_size = 50  # 50 for adam, 300 for lbfgs
    bo_sub_sample_batch_size = 1000  # 2000 for hpob, 300 for synthetic 4, 1000 for synthetic 5
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

    '''
    # ground truth for synthetic 4
    gt_gp_distribution_params = {
        'constant': (1.0, 1.0),
        'lengthscale': (10.0, 30.0),
        'signal_variance': (1.0, 1.0),
        'noise_variance': (10.0, 100000.0)
    }
    '''
    gt_gp_distribution_params = None

    kernel_type = kernel_list[0]
    # construct the jax random key
    key = jax.random.PRNGKey(0)

    # group_id = 'split_hpob_pos_1_50'
    group_id = 'split_synthetic_2_50'

    dir_path = os.path.join('results', 'test_hyperbo_plus_split', group_id)

    gamma_nll_train_list = []
    gamma_nll_test_list = []
    hyperbo_nll_train_list = []
    hyperbo_nll_test_list = []
    for dataset_id in setup_b_id_list:
        nll_results_train_id = np.load(
            os.path.join(dir_path, 'split_eval_nll_setup_b_train_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        gamma_nll_on_train_batches_i = np.mean(nll_results_train_id['gamma_nll_on_batches_i'])
        hyperbo_nll_on_train_batches_i = np.mean(nll_results_train_id['hyperbo_nll_on_batches_i'])
        gamma_nll_train_list.append(gamma_nll_on_train_batches_i)
        hyperbo_nll_train_list.append(hyperbo_nll_on_train_batches_i)

        nll_results_test_id = np.load(
            os.path.join(dir_path, 'split_eval_nll_setup_b_test_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        gamma_nll_on_test_batches_i = np.mean(nll_results_test_id['gamma_nll_on_batches_i'])
        hyperbo_nll_on_test_batches_i = np.mean(nll_results_test_id['hyperbo_nll_on_batches_i'])
        gamma_nll_test_list.append(gamma_nll_on_test_batches_i)
        hyperbo_nll_test_list.append(hyperbo_nll_on_test_batches_i)


    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(aggregation_dir_path):
        os.mkdir(aggregation_dir_path)

    x = np.arange(len(setup_b_id_list))  # the label locations
    width = 0.16 # the width of the bars
    gap = 0.02

    fig, ax = plt.subplots(figsize=(12, 3))
    rects1 = ax.bar(x - width * 1.5 - gap, gamma_nll_train_list, width, label='HyperBO+ (Train)')
    rects2 = ax.bar(x - width * 0.5 - gap, gamma_nll_test_list, width, label='HyperBO+ (Test)')
    rects3 = ax.bar(x + width * 0.5 + gap, hyperbo_nll_train_list, width, label='HyperBO (Train)')
    rects4 = ax.bar(x + width * 1.5 + gap, hyperbo_nll_test_list, width, label='HyperBO (Test)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Search Space ID')
    ax.set_ylabel('NLL')
    # ax.set_title('NLL')
    ax.set_xticks(x, setup_b_id_list, rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(aggregation_dir_path, 'split_nll_comparison.pdf'))
    plt.close(fig)
