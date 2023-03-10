import jax
from experiment_defs import *
import subprocess


if __name__ == '__main__':
    key = jax.random.PRNGKey(RANDOM_SEED)
    python_cmd = 'python'
    config_path = 'hyperbo/experiments_xmanager/hpl_bo_split_config.py'
    worker_path = 'hyperbo/experiments_xmanager/hpl_bo_split_worker.py'
    group_id = GROUP_ID
    train_id_list = TRAIN_ID_LIST
    test_id_list = TEST_ID_LIST
    setup_b_id_list = FULL_ID_LIST

    sub_process_i = subprocess.Popen([python_cmd, config_path, '--group_id', group_id])
    sub_process_i.wait()

    print('fit_single_gp')
    for train_id in train_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_gp_params_setup_a_id', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    for train_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_gp_params_setup_b_id', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()

    print('fit_two_step')
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_direct_hgp_two_step_setup_a', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_direct_hgp_two_step_setup_b', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    for train_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_direct_hgp_two_step_setup_b_leaveout', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_two_step_setup_a', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_two_step_setup_b', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    for train_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_hpl_hgp_two_step_setup_b_leaveout', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()

    print('fit_end_to_end')
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_end_to_end_setup_a', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_end_to_end_setup_b', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    for train_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_hpl_hgp_end_to_end_setup_b_leaveout', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_end_to_end_setup_a_no_init', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'fit_hpl_hgp_end_to_end_setup_b_no_init', '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()

    print('test_bo')
    for test_id in test_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'test_bo_setup_a_id', '--dataset_id', test_id,
                                          '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    for test_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'test_bo_setup_b_id', '--dataset_id', test_id,
                                          '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()

    print('eval_nll')
    for dataset_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_a_id', '--dataset_id', dataset_id,
                                          '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    for dataset_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_b_train_id', '--dataset_id', dataset_id,
                                          '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()
    for dataset_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_b_test_id', '--dataset_id', dataset_id,
                                          '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_i.wait()

    print('merge')
    new_key, key = jax.random.split(key)
    sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                      '--mode', 'merge',
                                      '--key_0',
                                      str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_i.wait()
