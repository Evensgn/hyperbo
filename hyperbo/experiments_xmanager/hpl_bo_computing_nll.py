from hyperbo.basics import definitions as defs
from hyperbo.basics import data_utils
from hyperbo.gp_utils import objectives as obj
import jax
import jax.numpy as jnp
import numpy as np
from multiprocessing import Pool
import os

from hpl_bo_utils import get_gp_params_samples_from_direct_hgp, get_gp_params_samples_from_hpl_hgp


def nll_on_sub_dataset(run_args):
    (gp_params, cov_func, mean_func, sub_dataset) = run_args
    dataset = {'only': sub_dataset}
    return obj.nll(
        mean_func,
        cov_func,
        gp_params,
        dataset,
        warp_func=None,
        exclude_aligned=True
    )


def nll_sub_dataset_level_with_gp(key, dataset, cov_func, mean_func, gp_params, eval_nll_batch_size,
                                  eval_nll_n_batches):
    # sub sample each sub dataset for large datasets
    new_key, key = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        new_key, dataset, eval_nll_batch_size)

    nll_loss_batches = []
    for k in range(eval_nll_n_batches):
        dataset_k = next(dataset_iter)
        nll_loss_list = []
        for sub_dataset in dataset_k.values():
            nll_i = nll_on_sub_dataset((gp_params, cov_func, mean_func, sub_dataset))
            nll_loss_list.append(nll_i)
        nll_loss = jnp.mean(jnp.array(nll_loss_list))
        nll_loss_batches.append(nll_loss)
    return nll_loss_batches


def nll_sub_dataset_level_with_gp_params_samples(key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples,
                                                 gp_params_samples, eval_nll_batch_size, eval_nll_n_batches,
                                                 lengthscale_eps=1e-6):
    # sub sample each sub dataset for large datasets
    key, subkey = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        subkey, dataset, eval_nll_batch_size)

    # sample gp params first
    constants, lengthscales, signal_variances, noise_variances = gp_params_samples
    lengthscales += lengthscale_eps

    nll_loss_batches = []
    for k in range(eval_nll_n_batches):
        dataset_k = next(dataset_iter)
        nll_loss_list = []
        for sub_dataset in dataset_k.values():
            task_list = []
            for i in range(n_nll_gp_params_samples):
                params_sample = defs.GPParams(
                    model={
                        'constant': constants[i],
                        'lengthscale': lengthscales[i],
                        'signal_variance': signal_variances[i],
                        'noise_variance': noise_variances[i]
                    }
                )
                task_list.append((params_sample, mean_func, cov_func, sub_dataset))
            if pool is not None:
                task_outputs = pool.map(nll_on_sub_dataset, task_list)
            else:
                task_outputs = []
                for i, task in enumerate(task_list):
                    task_outputs.append(nll_on_sub_dataset(task))
            objectives = jnp.array(task_outputs)
            nll_loss_sub_dataset = -(jax.scipy.special.logsumexp(-objectives, axis=0) - jnp.log(n_nll_gp_params_samples))
            nll_loss_list.append(nll_loss_sub_dataset)
        nll_loss = jnp.mean(jnp.array(nll_loss_list))
        nll_loss_batches.append(nll_loss)
    return nll_loss_batches


def split_eval_nll_setup_a_id(dir_path, key, dataset_id, dataset_func_combined, cov_func, mean_func, hand_hgp_params,
                              uniform_hgp_params, gt_hgp_params, nll_node_cpu_count, distribution_type,
                              dataset_dim_feature_values_path, method_name_list, setup_b_only_method_name_list,
                              n_nll_gp_params_samples, eval_nll_batch_size, eval_nll_n_batches):
    if nll_node_cpu_count is None or nll_node_cpu_count <= 1:
        pool = None
    else:
        pool = Pool(processes=nll_node_cpu_count)

    dataset = dataset_func_combined(dataset_id)
    n_dim = list(dataset.values())[0].x.shape[1]
    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[dataset_id]

    setup_a_method_name_list = [method_name for method_name in method_name_list
                                if method_name not in setup_b_only_method_name_list]

    results = {}

    if 'hand_hgp' in setup_a_method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, hand_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['hand_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'uniform_hgp' in setup_a_method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, uniform_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['uniform_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'gt_hgp' in setup_a_method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, gt_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['gt_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)

    # read fit direct hgp params
    if 'fit_direct_hgp' in setup_a_method_name_list:
        fit_direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_a.npy'),
                                        allow_pickle=True).item()['gp_distribution_params']
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, fit_direct_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['fit_direct_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in setup_a_method_name_list:
        hpl_hgp_end_to_end_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a.npy'),
                                            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_params, distribution_type,
                                                               n_nll_gp_params_samples, n_dim, dim_feature_value)
        results['hpl_hgp_end_to_end'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_end_to_end_from_scratch' in setup_a_method_name_list:
        hpl_hgp_end_to_end_from_scratch_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a_from_scratch.npy'),
            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_from_scratch_params,
                                                                distribution_type, n_nll_gp_params_samples, n_dim,
                                                                dim_feature_value)
        results['hpl_hgp_end_to_end_from_scratch'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_two_step' in setup_a_method_name_list:
        hpl_hgp_two_step_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_a.npy'),
                                          allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_two_step_params, distribution_type,
                                                                n_nll_gp_params_samples, n_dim, dim_feature_value)
        results['hpl_hgp_two_step'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)

    np.save(os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(dataset_id)), results)


def split_eval_nll_setup_b_id(dir_path, key, train_or_test, dataset_id, dataset_func_split, cov_func, mean_func,
                              hand_hgp_params, uniform_hgp_params, gt_hgp_params, nll_node_cpu_count, distribution_type,
                              dataset_dim_feature_values_path, method_name_list, n_nll_gp_params_samples,
                              eval_nll_batch_size, eval_nll_n_batches):
    if nll_node_cpu_count is None or nll_node_cpu_count <= 1:
        pool = None
    else:
        pool = Pool(processes=nll_node_cpu_count)

    if train_or_test == 'train':
        dataset = dataset_func_split(dataset_id)['train']  # only use train set
    elif train_or_test == 'test':
        dataset = dataset_func_split(dataset_id)['test']  # only use test set
    else:
        raise ValueError('Unknown train_or_test: {}'.format(train_or_test))
    n_dim = list(dataset.values())[0].x.shape[1]

    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[dataset_id]

    results = {}

    # read hyperbo gp params
    if 'hyperbo' in method_name_list:
        hyperbo_gp_parmas = np.load(
            os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        results['hyperbo'] = nll_sub_dataset_level_with_gp(key, dataset, cov_func, mean_func, hyperbo_gp_parmas,
                                                           eval_nll_batch_size, eval_nll_n_batches)

    if 'hand_hgp' in method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, hand_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['hand_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'uniform_hgp' in method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, uniform_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['uniform_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'gt_hgp' in method_name_list:
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, gt_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['gt_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)

    # read fit direct hgp params
    if 'fit_direct_hgp' in method_name_list:
        fit_direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b.npy'),
                                        allow_pickle=True).item()['gp_distribution_params']
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, fit_direct_hgp_params, distribution_type,
                                                                  n_nll_gp_params_samples, n_dim)
        results['fit_direct_hgp'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'fit_direct_hgp_leaveout' in method_name_list:
        fit_direct_hgp_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()['gp_distribution_params']
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_direct_hgp(new_key, fit_direct_hgp_leaveout_params,
                                                                  distribution_type, n_nll_gp_params_samples, n_dim)
        results['fit_direct_hgp_leaveout'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in method_name_list:
        hpl_hgp_end_to_end_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
                                            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_params, distribution_type,
                                                               n_nll_gp_params_samples, n_dim, dim_feature_value)
        results['hpl_hgp_end_to_end'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_end_to_end_leaveout' in method_name_list:
        hpl_hgp_end_to_end_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_leaveout_params,
                                                               distribution_type, n_nll_gp_params_samples, n_dim,
                                                               dim_feature_value)
        results['hpl_hgp_end_to_end_leaveout'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_end_to_end_from_scratch' in method_name_list:
        hpl_hgp_end_to_end_from_scratch_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_from_scratch_params,
                                                               distribution_type, n_nll_gp_params_samples, n_dim,
                                                               dim_feature_value)
        results['hpl_hgp_end_to_end_from_scratch'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_end_to_end_leaveout_from_scratch' in method_name_list:
        hpl_hgp_end_to_end_leaveout_from_scratch_params = np.load(
            os.path.join(dir_path,
                         'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch.npy'.format(dataset_id)),
            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_end_to_end_leaveout_from_scratch_params,
                                                               distribution_type, n_nll_gp_params_samples, n_dim,
                                                               dim_feature_value)
        results['hpl_hgp_end_to_end_leaveout_from_scratch'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_two_step' in method_name_list:
        hpl_hgp_two_step_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'),
                                          allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_two_step_params, distribution_type,
                                                               n_nll_gp_params_samples, n_dim, dim_feature_value)
        results['hpl_hgp_two_step'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    if 'hpl_hgp_two_step_leaveout' in method_name_list:
        hpl_hgp_two_step_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        new_key, key = jax.random.split(key, 2)
        gp_params_samples = get_gp_params_samples_from_hpl_hgp(new_key, hpl_hgp_two_step_leaveout_params,
                                                               distribution_type, n_nll_gp_params_samples, n_dim,
                                                               dim_feature_value)
        results['hpl_hgp_two_step_leaveout'] = nll_sub_dataset_level_with_gp_params_samples(
            key, pool, dataset, cov_func, mean_func, n_nll_gp_params_samples, gp_params_samples, eval_nll_batch_size,
            eval_nll_n_batches)
    np.save(os.path.join(dir_path, 'split_eval_nll_setup_b_{}_id_{}.npy'.format(train_or_test, dataset_id)), results)
