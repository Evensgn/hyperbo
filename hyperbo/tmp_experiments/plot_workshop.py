from matplotlib import pyplot as plt
import numpy as np
import argparse
import os


def plot_performance_curve(ax, label, regrets_mean, regrets_std, time_list, log_plot=False):
    err_low = [regrets_mean[i] - regrets_std[i] for i in range(len(regrets_mean))]
    err_high = [regrets_mean[i] + regrets_std[i] for i in range(len(regrets_mean))]

    if log_plot:
        line = ax.semilogy(time_list, regrets_mean, label=label)[0]
    else:
        line = ax.plot(time_list, regrets_mean, label=label)[0]
    ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


def plot_performance_curve_percentile(ax, label, regrets_all, time_list, log_plot=False):
    regrets_mean = np.mean(regrets_all, axis=0)
    err_low = np.percentile(regrets_all, 25, axis=0)
    err_high = np.percentile(regrets_all, 75, axis=0)

    if log_plot:
        line = ax.semilogy(time_list, regrets_mean, label=label)[0]
    else:
        line = ax.plot(time_list, regrets_mean, label=label)[0]
    ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


def plot_hyperbo_plus(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)

    # setup a
    results_a = results['setup_a']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('average best sample simple regret')
    plot_performance_curve_percentile(ax, 'fixed', results_a['bo_results_total']['fixed_regrets_all_list'], time_list)
    plot_performance_curve_percentile(ax, 'random', results_a['bo_results_total']['random_regrets_all_list'], time_list)
    plot_performance_curve_percentile(ax, 'hyperbo+', results_a['bo_results_total']['gamma_regrets_all_list'], time_list)
    ax.legend()
    fig.savefig(os.path.join(dir_path, 'setup_a_regret_vs_iteration.pdf'))
    plt.close(fig)

    # setup b
    results_b = results['setup_b']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('average best sample simple regret')
    plot_performance_curve_percentile(ax, 'fixed', results_b['bo_results_total']['fixed_regrets_all_list'],
                                      time_list)
    plot_performance_curve_percentile(ax, 'random', results_b['bo_results_total']['random_regrets_all_list'], time_list)
    plot_performance_curve_percentile(ax, 'hyperbo+', results_b['bo_results_total']['gamma_regrets_all_list'],
                                      time_list)
    plot_performance_curve_percentile(ax, 'hyperbo', results_b['bo_results_total']['hyperbo_regrets_all_list'],
                                      time_list)
    ax.legend()
    fig.savefig(os.path.join(dir_path, 'setup_b_regret_vs_iteration.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    results = np.load('results/test_hyperbo_plus_xxxx/results.npy', allow_pickle=True).item()
    plot_hyperbo_plus(results)

