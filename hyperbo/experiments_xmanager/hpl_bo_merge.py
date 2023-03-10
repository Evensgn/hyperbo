import jax.numpy as jnp
import numpy as np
import os
import plot


def split_merge(dir_path, group_id, configs):
    experiment_name = 'hpl_bo_split_group_id_{}_merge'.format(group_id)
    results = {}

    results['experiment_name'] = experiment_name

    # configs
    results['configs'] = configs

    train_id_list = configs['train_id_list']
    test_id_list = configs['test_id_list']
    setup_b_id_list = configs['setup_b_id_list']
    method_name_list = configs['method_name_list']
    setup_b_only_method_name_list = configs['setup_b_only_method_name_list']
    setup_a_method_name_list = [method_name for method_name in method_name_list
                                if method_name not in setup_b_only_method_name_list]
    ac_func_type_list = configs['ac_func_type_list']
    n_bo_runs = configs['n_bo_runs']
    eval_nll_n_batches = configs['eval_nll_n_batches']

    # setup a
    results['setup_a'] = {}
    results_a = results['setup_a']

    # fit gp parameters
    results_a['fit_gp_params'] = {}
    for train_id in train_id_list:
        results_a['fit_gp_params'][train_id] = np.load(
            os.path.join(dir_path, 'split_fit_gp_params_setup_a_id_{}.npy'.format(train_id)), allow_pickle=True).item()

    # read fit direct hgp params
    if 'fit_direct_hgp' in setup_a_method_name_list:
        results_a['fit_direct_hgp_params'] = np.load(
            os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_a.npy'),
            allow_pickle=True).item()['gp_distribution_params']

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in setup_a_method_name_list:
        results_a['hpl_hgp_end_to_end_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_from_scratch' in setup_a_method_name_list:
       results_a['hpl_hgp_end_to_end_from_scratch_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_a_from_scratch.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_two_step' in setup_a_method_name_list:
        results_a['hpl_hgp_two_step_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_a.npy'), allow_pickle=True).item()

    # run BO and compute NLL
    results_a['bo_results'] = {}
    results_a['bo_results_total'] = {}
    results_a['nll_results'] = {}

    bo_results_a = {}
    nll_results_a = {}
    for test_id in test_id_list:
        bo_results_a[test_id] = np.load(
            os.path.join(dir_path, 'split_test_bo_setup_a_id_{}.npy'.format(test_id)), allow_pickle=True).item()
    for dataset_id in setup_b_id_list:
        nll_results_a[dataset_id] = np.load(
            os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()

    for method_name in setup_a_method_name_list:
        # BO results
        results_a['bo_results'][method_name] = {}
        results_a['bo_results_total'][method_name] = {}
        for ac_func_type in ac_func_type_list:
            results_a['bo_results'][method_name][ac_func_type] = {}

            regrets_all_list = []

            for test_id in test_id_list:
                results_a['bo_results'][method_name][ac_func_type][test_id] = \
                    bo_results_a[test_id][ac_func_type][method_name]

            for i in range(n_bo_runs):
                regrets_all_list_i = []

                for test_id in test_id_list:
                    regrets_ij = results_a['bo_results'][method_name][ac_func_type][test_id]['regrets_list'][i]
                    regrets_all_list_i.append(regrets_ij)

                regrets_all_list.append(jnp.mean(jnp.array(regrets_all_list_i), axis=0))

            regrets_all_list = jnp.array(regrets_all_list)
            regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
            regrets_std_total = jnp.std(regrets_all_list, axis=0)

            results_a['bo_results_total'][method_name][ac_func_type] = {
                'regrets_all_list': regrets_all_list,
                'regrets_mean': regrets_mean_total,
                'regrets_std': regrets_std_total,
            }

        # NLL evaluations
        if method_name not in nll_results_a[train_id_list[0]]:
            continue

        nll_on_train_list = []
        for k in range(eval_nll_n_batches):
            nll_on_train_list.append([])
        for train_id in train_id_list:
            nll_on_train_batches_i = nll_results_a[train_id][method_name]
            for k in range(eval_nll_n_batches):
                nll_on_train_list[k].append(nll_on_train_batches_i[k])

        nll_on_test_list = []
        for k in range(eval_nll_n_batches):
            nll_on_test_list.append([])
        for test_id in test_id_list:
            fixed_nll_on_test_batches_i = nll_results_a[test_id][method_name]
            for k in range(eval_nll_n_batches):
                nll_on_test_list[k].append(fixed_nll_on_test_batches_i[k])

        nll_on_train_batches = []
        nll_on_test_batches = []
        for k in range(eval_nll_n_batches):
            nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
            nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

        results_a['nll_results'][method_name] = {
            'nll_on_train_mean': np.mean(nll_on_train_batches),
            'nll_on_train_std': np.std(nll_on_train_batches),
            'nll_on_test_mean': np.mean(nll_on_test_batches),
            'nll_on_test_std': np.std(nll_on_test_batches),
        }

    # setup b
    results['setup_b'] = {}
    results_b = results['setup_b']

    # fit gp parameters
    results_b['fit_gp_params'] = {}
    for dataset_id in setup_b_id_list:
        results_b['fit_gp_params'][dataset_id] = np.load(
            os.path.join(dir_path,
                         'split_fit_gp_params_setup_b_id_{}.npy'.format(dataset_id)), allow_pickle=True).item()

    # read fit direct hgp params
    if 'fit_direct_hgp' in method_name_list:
        results_b['fit_direct_hgp_params'] = np.load(
            os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b.npy'),
            allow_pickle=True).item()['gp_distribution_params']
    if 'fit_direct_hgp_leaveout' in method_name_list:
        results_b['fit_direct_hgp_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['fit_direct_hgp_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()['gp_distribution_params']

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in method_name_list:
        results_b['hpl_hgp_end_to_end_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_leaveout' in method_name_list:
        results_b['hpl_hgp_end_to_end_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_end_to_end_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_from_scratch' in method_name_list:
        results_b['hpl_hgp_end_to_end_from_scratch_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_leaveout_from_scratch' in method_name_list:
        results_b['hpl_hgp_end_to_end_leaveout_from_scratch_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_end_to_end_leaveout_from_scratch_params'][dataset_id] = np.load(
                os.path.join(dir_path,
                             'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch.npy'.format(dataset_id)),
                allow_pickle=True).item()
    if 'hpl_hgp_two_step' in method_name_list:
        results_b['hpl_hgp_two_step_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'), allow_pickle=True).item()
    if 'hpl_hgp_two_step_leaveout' in method_name_list:
        results_b['hpl_hgp_two_step_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_two_step_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()

    # run BO and compute NLL
    results_b['bo_results'] = {}
    results_b['bo_results_total'] = {}
    results_b['nll_results'] = {}

    bo_results_b = {}
    nll_results_b = {}
    for test_id in setup_b_id_list:
        bo_results_b[test_id] = np.load(
            os.path.join(dir_path, 'split_test_bo_setup_b_id_{}.npy'.format(test_id)), allow_pickle=True).item()
    for dataset_id in setup_b_id_list:
        nll_results_b[dataset_id] = {}
        nll_results_b[dataset_id]['train'] = np.load(
            os.path.join(dir_path, 'split_eval_nll_setup_b_train_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()
        nll_results_b[dataset_id]['test'] = np.load(
            os.path.join(dir_path, 'split_eval_nll_setup_b_test_id_{}.npy'.format(dataset_id)),
            allow_pickle=True).item()

    for method_name in method_name_list:
        # BO results
        results_b['bo_results'][method_name] = {}
        results_b['bo_results_total'][method_name] = {}
        for ac_func_type in ac_func_type_list:
            results_b['bo_results'][method_name][ac_func_type] = {}

            regrets_all_list = []

            for test_id in setup_b_id_list:
                results_b['bo_results'][method_name][ac_func_type][test_id] = \
                    bo_results_b[test_id][ac_func_type][method_name]

            for i in range(n_bo_runs):
                regrets_all_list_i = []

                for test_id in setup_b_id_list:
                    regrets_ij = results_b['bo_results'][method_name][ac_func_type][test_id]['regrets_list'][i]
                    regrets_all_list_i.append(regrets_ij)

                regrets_all_list.append(jnp.mean(jnp.array(regrets_all_list_i), axis=0))

            regrets_all_list = jnp.array(regrets_all_list)
            regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
            regrets_std_total = jnp.std(regrets_all_list, axis=0)

            results_b['bo_results_total'][method_name][ac_func_type] = {
                'regrets_all_list': regrets_all_list,
                'regrets_mean': regrets_mean_total,
                'regrets_std': regrets_std_total,
            }

        # NLL evaluations
        if method_name not in nll_results_b[setup_b_id_list[0]]['train']:
            continue

        nll_on_train_list = []
        for k in range(eval_nll_n_batches):
            nll_on_train_list.append([])
        for train_id in setup_b_id_list:
            nll_on_train_batches_i = nll_results_b[train_id]['train'][method_name]
            for k in range(eval_nll_n_batches):
                nll_on_train_list[k].append(nll_on_train_batches_i[k])

        nll_on_test_list = []
        for k in range(eval_nll_n_batches):
            nll_on_test_list.append([])
        for test_id in setup_b_id_list:
            fixed_nll_on_test_batches_i = nll_results_b[test_id]['test'][method_name]
            for k in range(eval_nll_n_batches):
                nll_on_test_list[k].append(fixed_nll_on_test_batches_i[k])

        nll_on_train_batches = []
        nll_on_test_batches = []
        for k in range(eval_nll_n_batches):
            nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
            nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

        results_a['nll_results'][method_name] = {
            'nll_on_train_mean': np.mean(nll_on_train_batches),
            'nll_on_train_std': np.std(nll_on_train_batches),
            'nll_on_test_mean': np.mean(nll_on_test_batches),
            'nll_on_test_std': np.std(nll_on_test_batches),
        }

    # save all results
    merge_path = os.path.join(dir_path, 'merge')
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    np.save(os.path.join(merge_path, 'results.npy'), results)

    # generate plots
    # plot.plot_hyperbo_plus(results)

    # write part of results to text file
    with open(os.path.join(merge_path, 'results.txt'), 'w') as f:
        f.write(str(results))

    print('done.')
