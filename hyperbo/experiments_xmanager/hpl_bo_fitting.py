from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import gp_added_v3
from hyperbo.gp_utils import utils
import jax
import jax.numpy as jnp
import numpy as np
import os
from tensorflow_probability.substrates.jax.distributions import Gamma, LogNormal
import optax


def fit_gp_params(key, dataset, cov_func, mean_func, objective, opt_method, fit_gp_maxiter, fit_gp_batch_size,
                  fit_gp_adam_learning_rate):
    n_dim = list(dataset.values())[0].x.shape[1]

    # minimize nll
    init_params = defs.GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method': opt_method,
            'maxiter': fit_gp_maxiter,
            'logging_interval': 10,
            'objective': objective,
            'batch_size': fit_gp_batch_size,
            'learning_rate': fit_gp_adam_learning_rate,
        })

    warp_func = utils.single_gp_default_warp_func

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    init_key, key = jax.random.split(key)

    model.initialize_params(init_key)

    inferred_params, init_nll, inferred_nll = model.train_return_loss()

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, params_utils.retrieve_params(inferred_params, param_keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))

    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def split_fit_gp_params_id(dir_path, key, setup, train_id, dataset_func_combined, dataset_func_split,
                           cov_func, mean_func, objective, opt_method, fit_gp_maxiter, fit_gp_batch_size,
                           fit_gp_adam_learning_rate):
    if setup == 'a':
        dataset = dataset_func_combined(train_id)
    elif setup == 'b':
        dataset = dataset_func_split(train_id)['train']  # only use training set
    else:
        raise ValueError('setup = {} not supported'.format(setup))
    new_key, key = jax.random.split(key)
    gp_params, nll_logs = fit_gp_params(new_key, dataset, cov_func, mean_func, objective, opt_method, fit_gp_maxiter,
                                        fit_gp_batch_size, fit_gp_adam_learning_rate)
    results = {'gp_params': gp_params, 'nll_logs': nll_logs}
    np.save(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)), results)


def split_fit_direct_hgp_two_step(dir_path, setup, train_id_list, leaveout_id=None):
    constant_list = []
    lengthscale_list = []
    signal_variance_list = []
    noise_variance_list = []

    results = {}

    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    for train_id in train_id_list:
        gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)),
                            allow_pickle=True).item()['gp_params']
        constant_list.append(gp_params['constant'])
        lengthscale_list += list(gp_params['lengthscale'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])

    gp_distribution_params = {}
    gp_distribution_params['constant'] = utils.normal_param_from_samples(np.array(constant_list))
    gp_distribution_params['lengthscale'] = utils.gamma_param_from_samples(np.array(lengthscale_list))
    gp_distribution_params['signal_variance'] = utils.gamma_param_from_samples(np.array(signal_variance_list))
    gp_distribution_params['noise_variance'] = utils.gamma_param_from_samples(np.array(noise_variance_list))

    results['gp_distribution_params'] = gp_distribution_params
    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_{}.npy'.format(setup))
    np.save(save_file_name, results)


def split_fit_hpl_hgp_two_step(dir_path, key, setup, train_id_list, fit_two_step_maxiter, fit_two_step_learning_rate,
                               distribution_type, dataset_dim_feature_values_path, leaveout_id=None):
    dim_feature_values = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()
    model_params = {}

    constant_list = []
    signal_variance_list = []
    noise_variance_list = []

    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    model_params['search_space_params'] = {}
    fit_gp_results = {}
    for train_id in train_id_list:
        gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)),
                            allow_pickle=True).item()['gp_params']
        fit_gp_results[train_id] = gp_params
        model_params['search_space_params'][train_id] = gp_params
        constant_list.append(gp_params['constant'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])

    # fit mlp for lengthscale
    lengthscale_dist_mlp_features = (2,)
    new_key, key = jax.random.split(key)
    init_val = jnp.ones((0, 4), jnp.float32)
    lengthscale_dist_mlp_params = bf.MLP(lengthscale_dist_mlp_features).init(new_key, init_val)['params']

    # optimization with adam
    optimizer = optax.adam(fit_two_step_learning_rate)
    opt_state = optimizer.init(lengthscale_dist_mlp_params)

    for iter in range(fit_two_step_maxiter):
        def loss_func(lengthscale_dist_mlp_params):
            lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
            loss = 0.
            for train_id in train_id_list:
                gp_params = fit_gp_results[train_id]
                dim_feature_value = dim_feature_values[train_id]
                lengthscale = gp_params['lengthscale']
                lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params},
                                                                  dim_feature_value)
                for dim in range(len(lengthscale)):
                    if distribution_type == 'gamma':
                        lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
                        lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                        gamma_dist = Gamma(lengthscale_a, rate=lengthscale_b)
                        loss += -gamma_dist.log_prob(lengthscale[dim])
                    elif distribution_type == 'lognormal':
                        lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
                        lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], \
                                                            lengthscale_dist_params_dim[1]
                        lognormal_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
                        loss += -lognormal_dist.log_prob(lengthscale[dim])
            return loss

        current_loss, grad = jax.value_and_grad(loss_func)(lengthscale_dist_mlp_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        lengthscale_dist_mlp_params = optax.apply_updates(lengthscale_dist_mlp_params, updates)
        print('iter:', iter, ', loss:', current_loss)

    if distribution_type == 'gamma':
        print('lengthscale_gamma_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_gamma_mlp_params'] = lengthscale_dist_mlp_params
    elif distribution_type == 'lognormal':
        print('lengthscale_lognormal_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_lognormal_mlp_params'] = lengthscale_dist_mlp_params

    # fit other parameters
    constant_list = jnp.array(constant_list)
    constant_mu, constant_sigma = utils.normal_param_from_samples(constant_list)
    model_params['constant_normal_params'] = (constant_mu, constant_sigma)
    print('constant: Normal(mu={}, sigma={})'.format(constant_mu, constant_sigma))

    signal_variance_list = jnp.array(signal_variance_list)
    if distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = utils.gamma_param_from_samples(signal_variance_list)
        model_params['signal_variance_gamma_params'] = (signal_variance_a, signal_variance_b)
        print('signal_variance: Gamma(alpha={}, beta={})'.format(signal_variance_a, signal_variance_b))
    elif distribution_type == 'lognormal':
        signal_variance_mu, signal_variance_sigma = utils.lognormal_param_from_samples(signal_variance_list)
        model_params['signal_variance_lognormal_params'] = (signal_variance_mu, signal_variance_sigma)
        print('signal_variance: LogNormal(mu={}, sigma={})'.format(signal_variance_mu, signal_variance_sigma))

    noise_variance_list = jnp.array(noise_variance_list)
    if distribution_type == 'gamma':
        noise_variance_a, noise_variance_b = utils.gamma_param_from_samples(noise_variance_list)
        model_params['noise_variance_gamma_params'] = (noise_variance_a, noise_variance_b)
        print('noise_variance: Gamma(alpha={}, beta={})'.format(noise_variance_a, noise_variance_b))
    elif distribution_type == 'lognormal':
        noise_variance_mu, noise_variance_sigma = utils.lognormal_param_from_samples(noise_variance_list)
        model_params['noise_variance_lognormal_params'] = (noise_variance_mu, noise_variance_sigma)
        print('noise_variance: LogNormal(mu={}, sigma={})'.format(noise_variance_mu, noise_variance_sigma))

    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_{}.npy'.format(setup))
    np.save(save_file_name, model_params)


def fit_hpl_hgp_end_to_end(key, super_dataset, dim_feature_values, cov_func, mean_func, objective, opt_method,
                           fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                           init_params_value):
    # minimize nll
    init_params = defs.GPParams(
        model = {},
        config = {
            'method': opt_method,
            'maxiter': fit_hgp_maxiter,
            'logging_interval': 10,
            'objective': objective,
            'batch_size': fit_hgp_batch_size,
            'learning_rate': fit_hgp_adam_learning_rate,
            'distribution_type': distribution_type,
        })

    single_gp_warp_func = utils.single_gp_default_warp_func
    single_gp_inverse_warp_func = utils.single_gp_default_inverse_warp_func
    hgp_warp_func, hgp_inverse_warp_func = utils.get_e2e_v3_warp_func(
        distribution_type=distribution_type,
        single_gp_warp_func=single_gp_warp_func,
        single_gp_inverse_warp_func=single_gp_inverse_warp_func,
    )

    init_params.config['lengthscale_{}_mlp_features'.format(distribution_type)] = (2,)
    if init_params_value is None:
        # initialize constant, signal variance, and noise variance distribution parameters
        new_key, key = jax.random.split(key)
        init_params.model['constant_normal_params'] = jax.random.normal(new_key, (2,))
        new_key, key = jax.random.split(key)
        init_params.model['signal_variance_{}_params'.format(distribution_type)] = jax.random.normal(new_key, (2,))
        new_key, key = jax.random.split(key)
        init_params.model['noise_variance_{}_params'.format(distribution_type)] = jax.random.normal(new_key, (2,))

        # initialize per-search-space gp parameters
        init_params.model['search_space_params'] = {}
        for dataset_id, dataset in super_dataset.items():
            n_dim = list(dataset.values())[0].x.shape[1]
            init_params.model['search_space_params'][dataset_id] = {
                'constant': 1.0,
                'lengthscale': jnp.array([1.0] * n_dim),
                'signal_variance': 0.,
                'noise_variance': -4.,
            }

        # initialize the lengthscale distribution mlp
        new_key, key = jax.random.split(key)
        init_val = jnp.ones((0, 4), jnp.float32)
        init_params.model['lengthscale_{}_mlp_params'.format(distribution_type)] = \
            bf.MLP(init_params.config['lengthscale_{}_mlp_features'.format(distribution_type)]).init(
            new_key, init_val)['params']
    else:
        init_params_value = utils.apply_warp_func(init_params_value, hgp_inverse_warp_func)
        init_params.model = init_params_value

    model = gp_added_v3.HGP_E2E_v3(
        super_dataset=super_dataset,
        dim_feature_values=dim_feature_values,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        hgp_warp_func=hgp_warp_func,
        single_gp_warp_func=single_gp_warp_func,
    )

    init_key, key = jax.random.split(key)

    model.initialize_params(init_key)

    init_nll = model.neg_log_marginal_likelihood_hgp()
    new_key, key = jax.random.split(key)
    inferred_params = model.train(key=new_key)
    inferred_nll = model.neg_log_marginal_likelihood_hgp()

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, params_utils.retrieve_params(inferred_params, param_keys, warp_func=hgp_warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))
    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def split_fit_hpl_hgp_end_to_end(dir_path, key, setup, train_id_list, dataset_func_combined, dataset_func_split,
                                 dataset_dim_feature_values_path, cov_func, mean_func, objective, opt_method,
                                 fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                 use_init_params_value=True, leaveout_id=None):
    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    super_dataset = {}
    for train_id in train_id_list:
        if setup == 'a':
            dataset = dataset_func_combined(train_id)
        elif setup == 'b':
            dataset = dataset_func_split(train_id)['train']  # only use training set
        else:
            raise ValueError('setup = {} not supported'.format(setup))
        super_dataset[train_id] = dataset

    dim_feature_values = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()
    if use_init_params_value:
        if leaveout_id is not None:
            init_params_value_path = os.path.join(
                dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
        else:
            init_params_value_path = os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_{}.npy'.format(setup))
        init_params_value = np.load(init_params_value_path, allow_pickle=True).item()
    else:
        init_params_value = None

    new_key, key = jax.random.split(key)
    gp_params, nll_logs = fit_hpl_hgp_end_to_end(new_key, super_dataset, dim_feature_values, cov_func, mean_func,
                                                 objective, opt_method, fit_hgp_maxiter, fit_hgp_batch_size,
                                                 fit_hgp_adam_learning_rate, distribution_type, init_params_value)
    results = {'gp_params': gp_params, 'nll_logs': nll_logs}
    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_{}.npy'.format(setup))
    if not use_init_params_value:
        save_file_name = save_file_name.replace('.npy', '_from_scratch.npy')
    np.save(save_file_name, results)
