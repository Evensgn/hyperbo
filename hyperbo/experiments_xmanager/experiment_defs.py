GROUP_ID = 'hpl_bo_new_full_hpob_budget_100_0'
RANDOM_SEED = 0

IS_HPOB = True

HPOB_TRAIN_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
HPOB_TEST_ID_LIST = ['6794', '7607', '7609', '5889']
HPOB_FULL_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                     '6794', '7607', '7609', '5889']

SYNTHETIC_TRAIN_ID_LIST = [str(x) for x in range(16)]
SYNTHETIC_TEST_ID_LIST = [str(x) for x in range(16, 20)]
SYNTHETIC_FULL_ID_LIST = [str(x) for x in range(20)]
'''
SYNTHETIC_TRAIN_ID_LIST = ['0', '1', '2']
SYNTHETIC_TEST_ID_LIST = ['3', '4']
SYNTHETIC_FULL_ID_LIST = ['0', '1', '2']
'''

if IS_HPOB:
    TRAIN_ID_LIST = HPOB_TRAIN_ID_LIST
    TEST_ID_LIST = HPOB_TEST_ID_LIST
    FULL_ID_LIST = HPOB_FULL_ID_LIST
else:
    TRAIN_ID_LIST = SYNTHETIC_TRAIN_ID_LIST
    TEST_ID_LIST = SYNTHETIC_TEST_ID_LIST
    FULL_ID_LIST = SYNTHETIC_FULL_ID_LIST

IS_GCP = True

if IS_GCP:
    HPOB_DATA_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/hpob-data'
    SYNTHETIC_DATA_PTH = '/gcs/bayesopt/zfan/bayesopt_datasets/synthetic_data/dataset_4.npy'
    RESULTS_DIR = '/gcs/bayesopt/zfan/bayesopt_results'
    HPOB_DATA_ANALYSIS_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/hpob-data/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = '/gcs/bayesopt/zfan/bayesopt_datasets/synthetic_data/data_analysis.npy'
    FITTING_NODE_CPU_COUNT = 8
    FITTING_E2E_NODE_CPU_COUNT = 16
    BO_NODE_CPU_COUNT = 60
    NLL_NODE_CPU_COUNT = 60
    BASIC_CPU_COUNT = 4
else:
    HPOB_DATA_PATH = '../hpob-data'
    SYNTHETIC_DATA_PTH = '../synthetic_data/dataset_4.npy'
    RESULTS_DIR = '../results'
    HPOB_DATA_ANALYSIS_PATH = '../hpob-data/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = '../synthetic_data/data_analysis.npy'
    FITTING_NODE_CPU_COUNT = 4
    FITTING_E2E_NODE_CPU_COUNT = 4
    BO_NODE_CPU_COUNT = 4
    NLL_NODE_CPU_COUNT = 4
