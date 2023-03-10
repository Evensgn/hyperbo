from hyperbo.basics import definitions as defs
from hyperbo.basics import data_utils
from hyperbo.bo_utils import bayesopt
import hyperbo.bo_utils.acfun as acfun
import jax
import jax.numpy as jnp
import numpy as np
import os
from pathos.multiprocessing import ProcessingPool


from hpl_bo_utils import get_gp_params_samples_from_direct_hgp, get_gp_params_samples_from_hpl_hgp


def run_bo_with_gp(cov_func, mean_func, gp_params, dataset, sub_dataset_key, queried_sub_dataset, ac_func, budget):
    observations, _, _ = bayesopt.run_synthetic(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=gp_params,
        warp_func=None,
        ac_func=ac_func,
        iters=budget,
    )
    return observations


def run_bo_with_gp_params_samples(cov_func, mean_func, n_dim, gp_params_samples, n_bo_gp_params_samples, dataset,
                                  sub_dataset_key, queried_sub_dataset, ac_func, budget, padding_len):
    observations = bayesopt.run_bo_with_gp_params_samples(
        n_dim=n_dim,
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        gp_params_samples=gp_params_samples,
        ac_func=ac_func,
        iters=budget,
        n_bo_gp_params_samples=n_bo_gp_params_samples,
        padding_len=padding_len,
    )
    return observations


def run_bo(run_args):
    (cov_func, mean_func, n_dim, gp_params, gp_params_samples, n_bo_gp_params_samples, queried_sub_dataset,
     init_indices, ac_func, budget, padding_len, method_name) = run_args

    sub_dataset_key = 'history'
    if init_indices is None:
        init_sub_dataset = defs.SubDataset(x=jnp.empty((0, n_dim)), y=jnp.empty((0, 1)))
    else:
        history_x = queried_sub_dataset.x[init_indices, :]
        history_y = queried_sub_dataset.y[init_indices, :]
        init_sub_dataset = defs.SubDataset(x=history_x, y=history_y)
    dataset = {sub_dataset_key: init_sub_dataset}

    if method_name in ['random', 'hyperbo']:
        observations = run_bo_with_gp(
            cov_func=cov_func,
            mean_func=mean_func,
            gp_params=gp_params,
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            budget=budget,
        )
    else:
        observations = run_bo_with_gp_params_samples(
            cov_func=cov_func,
            mean_func=mean_func,
            n_dim=n_dim,
            gp_params_samples=gp_params_samples,
            n_bo_gp_params_samples=n_bo_gp_params_samples,
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            budget=budget,
            padding_len=padding_len,
        )

    max_f = jnp.max(queried_sub_dataset.y)
    min_f = jnp.min(queried_sub_dataset.y)
    regrets = []
    max_y = -jnp.inf
    for y in observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        regrets.append((max_f - max_y) / (max_f - min_f))
    return regrets


def test_bo(key, pool, dataset, cov_func, mean_func, n_init_obs, init_indices_values, budget, n_bo_runs,
            n_bo_gp_params_samples, ac_func, hand_hgp_params, uniform_hgp_params, gt_hgp_params, hyperbo_params,
            fit_direct_hgp_params, fit_direct_hgp_leaveout_params, hpl_hgp_end_to_end_params,
            hpl_hgp_end_to_end_leaveout_params, hpl_hgp_end_to_end_from_scratch_params,
            hpl_hgp_end_to_end_leaveout_from_scratch_params, hpl_hgp_two_step_params, hpl_hgp_two_step_leaveout_params,
            bo_sub_sample_batch_size, distribution_type, dim_feature_value, method_name_list):
    n_dim = list(dataset.values())[0].x.shape[1]

    if bo_sub_sample_batch_size is not None:
        # sub sample each sub dataset for large datasets
        key, subkey = jax.random.split(key, 2)
        dataset_iter = data_utils.sub_sample_dataset_iterator(
            subkey, dataset, bo_sub_sample_batch_size)
        dataset = next(dataset_iter)
    else:
        '''
        # set the sub-sample size to the largest sub-dataset length
        for sub_dataset in dataset.values():
            if bo_sub_sample_batch_size is None or sub_dataset.x.shape[0] > bo_sub_sample_batch_size:
                bo_sub_sample_batch_size = sub_dataset.x.shape[0]
        '''

    # sample init_indices
    init_indices_map = {}
    for sub_dataset_key, sub_dataset in dataset.items():
        init_indices_map[sub_dataset_key] = {}
        for i in range(n_bo_runs):
            if n_init_obs == 0:
                init_indices_map[sub_dataset_key][i] = None
            else:
                if init_indices_values and 'test{}'.format(i) in init_indices_values[sub_dataset_key] and \
                        len(init_indices_values[sub_dataset_key]['test{}'.format(i)]) == n_init_obs:
                    init_indices_map[sub_dataset_key][i] = init_indices_values[sub_dataset_key]['test{}'.format(i)]
                else:
                    new_key, key = jax.random.split(key)
                    init_indices_map[sub_dataset_key][i] = jax.random.choice(
                        new_key, sub_dataset.x.shape[0], shape=(n_init_obs,), replace=False)

    # sample gp_params_samples and construct task list
    task_list = []
    for method_name in method_name_list:
        pass_ac_func = ac_func
        if method_name in ['random', 'hyperbo']:
            gp_params_samples_uncut = None
            if method_name == 'random':
                placeholder_params = defs.GPParams(
                    model={
                        'constant': 1.0,
                        'lengthscale': jnp.array([1.0] * n_dim),
                        'signal_variance': 1.0,
                        'noise_variance': 1e-6,
                    }
                )
                pass_ac_func = acfun.rand
                gp_params = placeholder_params
            elif method_name == 'hyperbo':
                gp_params = hyperbo_params
            else:
                raise ValueError('Unknown method name: {}'.format(method_name))
        elif method_name in ['hand_hgp', 'uniform_hgp', 'gt_hgp', 'fit_direct_hgp', 'fit_direct_hgp_leaveout']:
            gp_params = None
            if method_name == 'hand_hgp':
                direct_hgp_params = hand_hgp_params
            elif method_name == 'uniform_hgp':
                direct_hgp_params = uniform_hgp_params
            elif method_name == 'gt_hgp':
                direct_hgp_params = gt_hgp_params
            elif method_name == 'fit_direct_hgp':
                direct_hgp_params = fit_direct_hgp_params
            elif method_name == 'fit_direct_hgp_leaveout':
                direct_hgp_params = fit_direct_hgp_leaveout_params
            else:
                raise ValueError('Unknown method name: {}'.format(method_name))

            new_key, key = jax.random.split(key)
            gp_params_samples_uncut = get_gp_params_samples_from_direct_hgp(
                new_key, direct_hgp_params, distribution_type, n_bo_runs * n_bo_gp_params_samples, n_dim)
        elif method_name in ['hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_leaveout', 'hpl_hgp_end_to_end_from_scratch',
                             'hpl_hgp_end_to_end_leaveout_from_scratch', 'hpl_hgp_two_step',
                             'hpl_hgp_two_step_leaveout']:
            gp_params = None
            if method_name == 'hpl_hgp_end_to_end':
                hpl_hgp_params = hpl_hgp_end_to_end_params
            elif method_name == 'hpl_hgp_end_to_end_leaveout':
                hpl_hgp_params = hpl_hgp_end_to_end_leaveout_params
            elif method_name == 'hpl_hgp_end_to_end_from_scratch':
                hpl_hgp_params = hpl_hgp_end_to_end_from_scratch_params
            elif method_name == 'hpl_hgp_end_to_end_leaveout_from_scratch':
                hpl_hgp_params = hpl_hgp_end_to_end_leaveout_from_scratch_params
            elif method_name == 'hpl_hgp_two_step':
                hpl_hgp_params = hpl_hgp_two_step_params
            elif method_name == 'hpl_hgp_two_step_leaveout':
                hpl_hgp_params = hpl_hgp_two_step_leaveout_params
            else:
                raise ValueError('Unknown method name: {}'.format(method_name))

            new_key, key = jax.random.split(key)
            gp_params_samples_uncut = get_gp_params_samples_from_hpl_hgp(
                new_key, hpl_hgp_params, distribution_type, n_bo_runs * n_bo_gp_params_samples, n_dim, dim_feature_value)
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))

        for i in range(n_bo_runs):
            if gp_params_samples_uncut is not None:
                constants, lengthscales, signal_variances, noise_variances = gp_params_samples_uncut
                constants_cut = constants[i * n_bo_gp_params_samples:(i + 1) * n_bo_gp_params_samples]
                lengthscales_cut = lengthscales[i * n_bo_gp_params_samples:(i + 1) * n_bo_gp_params_samples]
                signal_variances_cut = signal_variances[i * n_bo_gp_params_samples:(i + 1) * n_bo_gp_params_samples]
                noise_variances_cut = noise_variances[i * n_bo_gp_params_samples:(i + 1) * n_bo_gp_params_samples]
                gp_params_samples = (constants_cut, lengthscales_cut, signal_variances_cut, noise_variances_cut)
            else:
                gp_params_samples = None
            for sub_dataset_key, sub_dataset in dataset.items():
                init_indices = init_indices_map[sub_dataset_key][i]
                new_key, key = jax.random.split(key)
                task_list.append((cov_func, mean_func, n_dim, gp_params, gp_params_samples, n_bo_gp_params_samples,
                                  sub_dataset, init_indices, pass_ac_func, budget, bo_sub_sample_batch_size,
                                  method_name))
    print('task_list constructed, number of tasks: {}'.format(len(task_list)))

    if pool is not None:
        print('using pool')
        task_outputs = pool.map(run_bo, task_list)
    else:
        task_outputs = []
        for i, task in enumerate(task_list):
            print('task number {}'.format(i))
            task_outputs.append(run_bo(task))

    print('task_outputs computed')

    n_sub_datasets = len(dataset)
    results = {}
    for i, method_name in enumerate(method_name_list):
        regrets_list = []
        for j in range(n_bo_runs):
            regrets_j_list = task_outputs[(i*n_bo_runs+j)*n_sub_datasets: (i*n_bo_runs+j+1)*n_sub_datasets]
            regrets_list.append(jnp.mean(jnp.array(regrets_j_list), axis=0))
        regrets_list = jnp.array(regrets_list)
        results[method_name] = {
            'regrets_list': regrets_list,
        }
    return results


def split_test_bo_setup_a_id(dir_path, key, test_id, dataset_func_combined, cov_func, mean_func, n_init_obs, budget,
                             n_bo_runs, n_bo_gp_params_samples, ac_func_type_list, hand_hgp_params, uniform_hgp_params,
                             gt_hgp_params, bo_sub_sample_batch_size, bo_node_cpu_count, distribution_type,
                             dataset_dim_feature_values_path, method_name_list, setup_b_only_method_name_list):
    results = {}

    if bo_node_cpu_count is None or bo_node_cpu_count <= 1:
        pool = None
    else:
        pool = ProcessingPool(bo_node_cpu_count)

    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[test_id]

    setup_a_method_name_list = [method_name for method_name in method_name_list
                                if method_name not in setup_b_only_method_name_list]

    # read fit direct hgp params
    if 'fit_direct_hgp' in setup_a_method_name_list:
        fit_direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_a.npy'),
                                        allow_pickle=True).item()['gp_distribution_params']
    else:
        fit_direct_hgp_params = None

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in setup_a_method_name_list:
        hpl_hgp_end_to_end_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a.npy'),
                                            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_params = None
    if 'hpl_hgp_end_to_end_from_scratch' in setup_a_method_name_list:
        hpl_hgp_end_to_end_from_scratch_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a_from_scratch.npy'),
            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_from_scratch_params = None
    if 'hpl_hgp_two_step' in setup_a_method_name_list:
        hpl_hgp_two_step_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_a.npy'),
                                          allow_pickle=True).item()
    else:
        hpl_hgp_two_step_params = None

    # set non-exist params to None
    hyperbo_params = None
    fit_direct_hgp_leaveout_params = None
    hpl_hgp_end_to_end_leaveout_params = None
    hpl_hgp_end_to_end_leaveout_from_scratch_params = None
    hpl_hgp_two_step_leaveout_params = None

    init_indices_values = None

    for ac_func_type in ac_func_type_list:
        print('ac_func_type = {}'.format(ac_func_type))
        if ac_func_type == 'ucb':
            ac_func = acfun.ucb
        elif ac_func_type == 'ei':
            ac_func = acfun.ei
        elif ac_func_type == 'pi':
            ac_func = acfun.pi
        else:
            raise ValueError('Unknown ac_func_type: {}'.format(ac_func_type))

        dataset = dataset_func_combined(test_id)

        new_key, key = jax.random.split(key)
        results_i = test_bo(
            key, pool, dataset, cov_func, mean_func, n_init_obs, init_indices_values, budget, n_bo_runs,
            n_bo_gp_params_samples, ac_func, hand_hgp_params, uniform_hgp_params, gt_hgp_params, hyperbo_params,
            fit_direct_hgp_params, fit_direct_hgp_leaveout_params, hpl_hgp_end_to_end_params,
            hpl_hgp_end_to_end_leaveout_params, hpl_hgp_end_to_end_from_scratch_params,
            hpl_hgp_end_to_end_leaveout_from_scratch_params, hpl_hgp_two_step_params, hpl_hgp_two_step_leaveout_params,
            bo_sub_sample_batch_size, distribution_type, dim_feature_value, setup_a_method_name_list,
        )
        results[ac_func_type] = results_i
    np.save(os.path.join(dir_path, 'split_test_bo_setup_a_id_{}.npy'.format(test_id)), results)


def split_test_bo_setup_b_id(dir_path, key, test_id, dataset_func_split, cov_func, mean_func, n_init_obs, budget,
                             n_bo_runs, n_bo_gp_params_samples, ac_func_type_list, hand_hgp_params, uniform_hgp_params,
                             gt_hgp_params, bo_sub_sample_batch_size, bo_node_cpu_count, distribution_type,
                             dataset_dim_feature_values_path, method_name_list):
    results = {}

    if bo_node_cpu_count is None or bo_node_cpu_count <= 1:
        pool = None
    else:
        pool = ProcessingPool(bo_node_cpu_count)

    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[test_id]

    # read hyperbo params
    if 'hyperbo' in method_name_list:
        hyperbo_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(test_id)),
                                         allow_pickle=True).item()['gp_params']
        hyperbo_params = defs.GPParams(model=hyperbo_params)
    else:
        hyperbo_params = None

    # read fit direct hgp params
    if 'fit_direct_hgp' in method_name_list:
        fit_direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b.npy'),
                                        allow_pickle=True).item()['gp_distribution_params']
    else:
        fit_direct_hgp_params = None
    if 'fit_direct_hgp_leaveout' in method_name_list:
        fit_direct_hgp_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(test_id)),
            allow_pickle=True).item()['gp_distribution_params']
    else:
        fit_direct_hgp_leaveout_params = None

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in method_name_list:
        hpl_hgp_end_to_end_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
                                            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_params = None
    if 'hpl_hgp_end_to_end_leaveout' in method_name_list:
        hpl_hgp_end_to_end_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(test_id)),
            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_leaveout_params = None
    if 'hpl_hgp_end_to_end_from_scratch' in method_name_list:
        hpl_hgp_end_to_end_from_scratch_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_from_scratch_params = None
    if 'hpl_hgp_end_to_end_leaveout_from_scratch' in method_name_list:
        hpl_hgp_end_to_end_leaveout_from_scratch_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch.npy'.format(test_id)),
            allow_pickle=True).item()['gp_params']
    else:
        hpl_hgp_end_to_end_leaveout_from_scratch_params = None
    if 'hpl_hgp_two_step' in method_name_list:
        hpl_hgp_two_step_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'),
                                          allow_pickle=True).item()
    else:
        hpl_hgp_two_step_params = None
    if 'hpl_hgp_two_step_leaveout' in method_name_list:
        hpl_hgp_two_step_leaveout_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(test_id)),
            allow_pickle=True).item()
    else:
        hpl_hgp_two_step_leaveout_params = None

    for ac_func_type in ac_func_type_list:
        if ac_func_type == 'ucb':
            ac_func = acfun.ucb
        elif ac_func_type == 'ei':
            ac_func = acfun.ei
        elif ac_func_type == 'pi':
            ac_func = acfun.pi
        else:
            raise ValueError('Unknown ac_func_type: {}'.format(ac_func_type))

        dataset_all = dataset_func_split(test_id)
        dataset = dataset_all['test']  # only use test set
        if 'init_indices' in dataset_all:
            init_indices_values = dataset_all['init_indices']
        else:
            init_indices_values = None

        new_key, key = jax.random.split(key)
        results_i = test_bo(
            key, pool, dataset, cov_func, mean_func, n_init_obs, init_indices_values, budget, n_bo_runs,
            n_bo_gp_params_samples, ac_func, hand_hgp_params, uniform_hgp_params, gt_hgp_params, hyperbo_params,
            fit_direct_hgp_params, fit_direct_hgp_leaveout_params, hpl_hgp_end_to_end_params,
            hpl_hgp_end_to_end_leaveout_params, hpl_hgp_end_to_end_from_scratch_params,
            hpl_hgp_end_to_end_leaveout_from_scratch_params, hpl_hgp_two_step_params, hpl_hgp_two_step_leaveout_params,
            bo_sub_sample_batch_size, distribution_type, dim_feature_value, method_name_list,
        )
        results[ac_func_type] = results_i
    np.save(os.path.join(dir_path, 'split_test_bo_setup_b_id_{}.npy'.format(test_id)), results)
