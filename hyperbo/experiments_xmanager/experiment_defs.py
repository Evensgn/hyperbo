GROUP_ID = 'split_hpob_e2e_v3_fitting_10000_and_bo'
RANDOM_SEED = 0

HPOB_TRAIN_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
HPOB_TEST_ID_LIST = ['6794', '7607', '7609', '5889']
HPOB_FULL_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                    '6794', '7607', '7609', '5889']

SYNTHETIC_TRAIN_ID_LIST = list(range(16))
SYNTHETIC_TEST_ID_LIST = list(range(16, 20))
SYNTHETIC_FULL_ID_LIST = list(range(20))

IS_GCP = False

if IS_GCP:
    HPOB_DATA_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/hpob-data'
    SYNTHETIC_DATA_PTH = '/gcs/bayesopt/zfan/bayesopt_datasets/synthetic_data/dataset_4.npy'
    RESULTS_DIR = '/gcs/bayesopt/zfan/bayesopt_results'
    HPOB_DATA_ANALYSIS_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/hpob-data/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/synthetic_data/data_analysis.npy'
    INIT_PARAMS_VALUE_SETUP_B_PATH = '/gcs/bayesopt/zfan/bayesopt_results/jump_start/model_params_two_step_setup_b.npy'
else:
    HPOB_DATA_PATH = '../hpob-data'
    SYNTHETIC_DATA_PTH = '../synthetic_data/dataset_4.npy'
    RESULTS_DIR = '../results'
    HPOB_DATA_ANALYSIS_PATH = '../hpob-data/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = '../synthetic_data/data_analysis.npy'
    INIT_PARAMS_VALUE_SETUP_B_PATH = '../results/mlp_lengthscale_lab/model_params_two_step_setup_b.npy'

