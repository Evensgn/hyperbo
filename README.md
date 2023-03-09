This repo is built based on the codebase of [HyperBO](https://github.com/google-research/hyperbo). We made several modifications to some original files to expand its utility and added new files under the `hyperbo/experiments` folder.

## Usage of HyperBO+

- Synthetic Super-dataset generation: see `hyperbo/experiments/synthetic_data_generation.py`. This script generates the Synthetic Super-dataset for the experiments of HyperBO+. Parameters of data-generation including the save path of the generated file are configured in the script.
- Run HyperBO+ experiments: see `hyperbo/experiments/test_hyperbo_plus_split_scheduler.py`. This script runs the experiments of HyperBO+. Hyper-parameters of the experiments are configured in the script. This file calls `hyperbo/experiments/test_hyperbo_plus_split_worker.py` using multiprocessing to run the experiments in parallel.
- Worker functions of HyperBO+ experiments: see `hyperbo/experiments/test_hyperbo_plus_split_worker.py`. This file contains the worker functions of the experiments of HyperBO+. The worker functions are called by `hyperbo/experiments/test_hyperbo_plus_split_scheduler.py` using multiprocessing.
- Plotting functions of BO performance curves: see `hyperbo/experiments/plot.py`.
- Test the asymptotic behavior of fitting a single GP: see `hyperbo/experiments/test_asymptotics.py`.
- Test the asymptotic behavior of the two-step pre-training of HyperBO+ on the Synthetic Super-dataset: see `hyperbo/experiments/test_hyperbo_plus_split_asymptotic_scheduler.py` and `hyperbo/experiments/test_hyperbo_plus_split_asymptotic_aggregator.py`.
- Visualize the NLL evaluations in Setup B per search space: see `hyperbo/experiments/test_hyperbo_plus_split_check_per_dataset.py`.
- Dataset loaders for the Synthetic Super-dataset and HPO-B Super-dataset: see `hyperbo/bo_utils/data.py`. Define the file path to the HPO-B Super-dataset in this file.

The remaining part of this README document is copied from the original repo except from slight modification to the installation instructions.

---

# HyperBO - Prior Discovery
A Jax/Flax codebase for prior discovery in meta Bayesian optimization.
The algorithm and analyses can be found in *[Pre-trained Gaussian processes for Bayesian optimization](https://arxiv.org/pdf/2109.08215.pdf)*. Slides are available [at this link](https://ziw.mit.edu/pub/hyperbo_slides.pdf) with [video at the AutoML Seminars](https://www.youtube.com/watch?v=cH4-hHXvO5c). 

Also see [GPax](https://github.com/google-research/gpax) for a more modular implementation of Gaussian processes used by HyperBO based on [Tensorflow Probability](https://www.tensorflow.org/probability) with Jax backend.

Disclaimer: This is not an officially supported Google product.

## Installation
We recommend using Python 3.7 for stability.

To install this codebase as a library inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install .
```

## Dataset
To download the dataset, please copy and paste the following link to your browser's address bar.
```
http://storage.googleapis.com/gresearch/pint/pd1.tar.gz
```
See pd1/README.txt for more information. The data is licensed under the CC-BY 4.0 license.

If you'd like to use the evaluations at each training step, the relevant columns of the data frame are
```
'valid/ce_loss'
'train/ce_loss',
'train/error_rate',
```
etc. They will hold arrays aligned with the global_step column that indicates what training step the measurement was taken at.

See the "best_\*" columns for the best measurement achieved over training.


## Usage
See tests.

## Citing
```
@article{wang2021hyperbo,
  title={Pre-training helps Bayesian optimization too},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2022}
}
```
