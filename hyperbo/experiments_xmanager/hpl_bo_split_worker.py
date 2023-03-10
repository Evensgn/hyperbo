import argparse
import os
import jax.numpy as jnp
import numpy as np

from experiment_defs import RESULTS_DIR
from hpl_bo_fitting import split_fit_gp_params_id, split_fit_direct_hgp_two_step, split_fit_hpl_hgp_two_step, \
    split_fit_hpl_hgp_end_to_end
from hpl_bo_running_bo import split_test_bo_setup_a_id, split_test_bo_setup_b_id
from hpl_bo_computing_nll import split_eval_nll_setup_a_id, split_eval_nll_setup_b_id
from hpl_bo_merge import split_merge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPL-BO split worker.')

    parser.add_argument('--group_id', default='split_0', type=str, help='split group id')
    parser.add_argument('--mode', default='', type=str, help='mode')
    parser.add_argument('--dataset_id', default='', type=str, help='dataset id')
    parser.add_argument('--key_0', default=0, type=int, help='key 0')
    parser.add_argument('--key_1', default=0, type=int, help='key 1')
    args = parser.parse_args()

    dir_path = os.path.join(RESULTS_DIR, 'hpl_bo_split', args.group_id)

    # construct the jax random key
    key = jnp.array([args.key_0, args.key_1], dtype=jnp.uint32)

    # read configs
    configs = np.load(os.path.join(dir_path, 'configs.npy'), allow_pickle=True).item()

    random_seed = configs['random_seed']
    train_id_list = configs['train_id_list']
    test_id_list = configs['test_id_list']
    setup_b_id_list = configs['setup_b_id_list']
    dataset_func_combined = configs['dataset_func_combined']
    dataset_func_split = configs['dataset_func_split']
    dataset_dim_feature_values_path = configs['dataset_dim_feature_values_path']
    extra_info = configs['extra_info']

    fit_gp_maxiter = configs['fit_gp_maxiter']
    fit_gp_batch_size = configs['fit_gp_batch_size']
    fit_gp_adam_learning_rate = configs['fit_gp_adam_learning_rate']

    fit_hgp_maxiter = configs['fit_hgp_maxiter']
    fit_hgp_batch_size = configs['fit_hgp_batch_size']
    fit_hgp_adam_learning_rate = configs['fit_hgp_adam_learning_rate']

    fit_two_step_maxiter = configs['fit_two_step_maxiter']
    fit_two_step_learning_rate = configs['fit_two_step_learning_rate']

    n_init_obs = configs['n_init_obs']
    budget = configs['budget']
    n_bo_runs = configs['n_bo_runs']
    n_bo_gp_params_samples = configs['n_bo_gp_params_samples']
    n_nll_gp_params_samples = configs['n_nll_gp_params_samples']
    bo_sub_sample_batch_size = configs['bo_sub_sample_batch_size']
    eval_nll_batch_size = configs['eval_nll_batch_size']
    eval_nll_n_batches = configs['eval_nll_n_batches']
    ac_func_type_list = configs['ac_func_type_list']

    hand_hgp_params = configs['hand_hgp_params']
    uniform_hgp_params = configs['uniform_hgp_params']
    gt_hgp_params = configs['gt_hgp_params']

    kernel_type = configs['kernel_type']
    mean_func = configs['mean_func']
    distribution_type = configs['distribution_type']

    fitting_node_cpu_count = configs['fitting_node_cpu_count']
    bo_node_cpu_count = configs['bo_node_cpu_count']
    nll_node_cpu_count = configs['nll_node_cpu_count']

    method_name_list = configs['method_name_list']
    setup_b_only_method_name_list = configs['setup_b_only_method_name_list']

    kernel_name, cov_func, gp_objective, hgp_objective, opt_method = kernel_type

    if args.mode == 'fit_gp_params_setup_a_id':
        split_fit_gp_params_id(dir_path, key, 'a', args.dataset_id, dataset_func_combined, dataset_func_split,
                               cov_func, mean_func, gp_objective, opt_method, fit_gp_maxiter, fit_gp_batch_size,
                               fit_gp_adam_learning_rate)
    elif args.mode == 'fit_gp_params_setup_b_id':
        split_fit_gp_params_id(dir_path, key, 'b', args.dataset_id, dataset_func_combined, dataset_func_split,
                               cov_func, mean_func, gp_objective, opt_method, fit_gp_maxiter, fit_gp_batch_size,
                               fit_gp_adam_learning_rate)

    elif args.mode == 'fit_direct_hgp_two_step_setup_a':
        split_fit_direct_hgp_two_step(dir_path, 'a', train_id_list)
    elif args.mode == 'fit_direct_hgp_two_step_setup_b':
        split_fit_direct_hgp_two_step(dir_path, 'b', setup_b_id_list)
    elif args.mode == 'fit_direct_hgp_two_step_setup_b_leaveout':
        split_fit_direct_hgp_two_step(dir_path, 'b', setup_b_id_list, leaveout_id=args.dataset_id)
    elif args.mode == 'fit_hpl_hgp_two_step_setup_a':
        split_fit_hpl_hgp_two_step(dir_path, key, 'a', train_id_list, fit_two_step_maxiter, fit_two_step_learning_rate,
                                   distribution_type, dataset_dim_feature_values_path)
    elif args.mode == 'fit_hpl_hgp_two_step_setup_b':
        split_fit_hpl_hgp_two_step(dir_path, key, 'b', setup_b_id_list, fit_two_step_maxiter,
                                   fit_two_step_learning_rate, distribution_type, dataset_dim_feature_values_path)
    elif args.mode == 'fit_hpl_hgp_two_step_setup_b_leaveout':
        split_fit_hpl_hgp_two_step(dir_path, key, 'b', setup_b_id_list, fit_two_step_maxiter,
                                   fit_two_step_learning_rate, distribution_type, dataset_dim_feature_values_path,
                                   leaveout_id=args.dataset_id)

    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_a':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'a', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, opt_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=True)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, opt_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=True)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b_leaveout':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, opt_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=True, leaveout_id=args.dataset_id)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_a_no_init':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'a', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, opt_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=False)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b_no_init':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, opt_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=False)

    elif args.mode == 'test_bo_setup_a_id':
        split_test_bo_setup_a_id(dir_path, key, args.dataset_id, dataset_func_combined, cov_func, mean_func, n_init_obs,
                                 budget, n_bo_runs, n_bo_gp_params_samples, ac_func_type_list, hand_hgp_params,
                                 uniform_hgp_params, gt_hgp_params, bo_sub_sample_batch_size, bo_node_cpu_count,
                                 distribution_type, dataset_dim_feature_values_path, method_name_list,
                                 setup_b_only_method_name_list)
    elif args.mode == 'test_bo_setup_b_id':
        split_test_bo_setup_b_id(dir_path, key, args.dataset_id, dataset_func_split, cov_func, mean_func, n_init_obs,
                                 budget, n_bo_runs, n_bo_gp_params_samples, ac_func_type_list, hand_hgp_params,
                                 uniform_hgp_params, gt_hgp_params, bo_sub_sample_batch_size, bo_node_cpu_count,
                                 distribution_type, dataset_dim_feature_values_path, method_name_list)

    elif args.mode == 'eval_nll_setup_a_id':
        split_eval_nll_setup_a_id(dir_path, key, args.dataset_id, dataset_func_combined, cov_func, mean_func,
                                  hand_hgp_params, uniform_hgp_params, gt_hgp_params, nll_node_cpu_count,
                                  distribution_type, dataset_dim_feature_values_path, method_name_list,
                                  setup_b_only_method_name_list, n_nll_gp_params_samples, eval_nll_batch_size,
                                  eval_nll_n_batches)
    elif args.mode == 'eval_nll_setup_b_train_id':
        split_eval_nll_setup_b_id(dir_path, key, 'train', args.dataset_id, dataset_func_split, cov_func, mean_func,
                                  hand_hgp_params, uniform_hgp_params, gt_hgp_params, nll_node_cpu_count,
                                  distribution_type, dataset_dim_feature_values_path, method_name_list,
                                  n_nll_gp_params_samples, eval_nll_batch_size, eval_nll_n_batches)
    elif args.mode == 'eval_nll_setup_b_test_id':
        split_eval_nll_setup_b_id(dir_path, key, 'test', args.dataset_id, dataset_func_split, cov_func, mean_func,
                                  hand_hgp_params, uniform_hgp_params, gt_hgp_params, nll_node_cpu_count,
                                  distribution_type, dataset_dim_feature_values_path, method_name_list,
                                  n_nll_gp_params_samples, eval_nll_batch_size, eval_nll_n_batches)

    elif args.mode == 'merge':
        split_merge(dir_path, args.group_id, configs)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))
