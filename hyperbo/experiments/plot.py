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


def plot_estimated_prior(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)
    for kernel in results['kernel_list']:
        kernel_results = results['kernel_results'][kernel]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('BO iteration')
        ax.set_ylabel('average best sample simple regret')
        plot_performance_curve(ax, 'groudtruth', kernel_results['regrets_mean_groundtruth'],
                               kernel_results['regrets_std_groundtruth'], time_list)
        plot_performance_curve(ax, 'inferred', kernel_results['regrets_mean_inferred'],
                               kernel_results['regrets_std_inferred'], time_list)
        plot_performance_curve(ax, 'random', kernel_results['regrets_mean_random'],
                                 kernel_results['regrets_std_random'], time_list)
        ax.legend()
        fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}.pdf'.format(kernel)))
        plt.close(fig)

        # visualize bo
        if results['visualize_bo']:
            visualize_bo_results = kernel_results['visualize_bo_results']
            n_visualize_grid_points = visualize_bo_results['n_visualize_grid_points']
            observations_groundtruth = visualize_bo_results['observations_groundtruth']
            observations_inferred = visualize_bo_results['observations_inferred']
            for i in range(results['budget']):
                # plot based on same observations
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_title('BO iteration = {} (same observations)'.format(i + 1))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.plot(visualize_bo_results['f_x'][:n_visualize_grid_points].squeeze(),
                        visualize_bo_results['f_y'][:n_visualize_grid_points].squeeze(), '--', label='f')
                ax.plot(visualize_bo_results['f_x'][n_visualize_grid_points:].squeeze(),
                        visualize_bo_results['f_y'][n_visualize_grid_points:].squeeze(), 'o', label='f_discrete')
                mean_groundtruth = visualize_bo_results['posterior_list'][i]['mean_groundtruth'].squeeze()
                std_groundtruth = visualize_bo_results['posterior_list'][i]['std_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_groundtruth[:n_visualize_grid_points], label='groundtruth')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] - std_groundtruth[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] + std_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_groundtruth[0][:i], observations_groundtruth[1][:i], 'o', color=line.get_color(), label='obs_gt')
                mean_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i]['mean_inferred_on_groundtruth'].squeeze()
                std_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i]['std_inferred_on_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_inferred_on_groundtruth[:n_visualize_grid_points], label='inferred (on obs_gt)')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] - std_inferred_on_groundtruth[:n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] + std_inferred_on_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.legend()
                fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}_same_obs_iteration_{}.pdf'.format(kernel, i)))
                plt.close(fig)

                # plot based on different observations
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_title('BO iteration = {} (different observations)'.format(i + 1))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.plot(visualize_bo_results['f_x'][:n_visualize_grid_points].squeeze(),
                        visualize_bo_results['f_y'][:n_visualize_grid_points].squeeze(), '--', label='f')
                ax.plot(visualize_bo_results['f_x'][n_visualize_grid_points:].squeeze(),
                        visualize_bo_results['f_y'][n_visualize_grid_points:].squeeze(), 'o', label='f_discrete')
                mean_groundtruth = visualize_bo_results['posterior_list'][i]['mean_groundtruth'].squeeze()
                std_groundtruth = visualize_bo_results['posterior_list'][i]['std_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_groundtruth[:n_visualize_grid_points], label='groundtruth')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] - std_groundtruth[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] + std_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_groundtruth[0][:i], observations_groundtruth[1][:i], 'o', color=line.get_color(),
                        label='obs_gt')
                mean_inferred_on_inferred = visualize_bo_results['posterior_list'][i][
                    'mean_inferred_on_inferred'].squeeze()
                std_inferred_on_inferred = visualize_bo_results['posterior_list'][i][
                    'std_inferred_on_inferred'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_inferred_on_inferred[:n_visualize_grid_points],
                               label='inferred (on obs_inf)')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_inferred_on_inferred[:n_visualize_grid_points] - std_inferred_on_inferred[
                                                                                         :n_visualize_grid_points],
                                mean_inferred_on_inferred[:n_visualize_grid_points] + std_inferred_on_inferred[
                                                                                         :n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_inferred[0][:i], observations_inferred[1][:i], 'o', color=line.get_color(),
                        label='obs_inf')
                ax.legend()
                fig.savefig(
                    os.path.join(dir_path, 'regret_vs_iteration_{}_different_obs_iteration_{}.pdf'.format(kernel, i)))
                plt.close(fig)


def plot_hyperbo_plus(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('average best sample simple regret')
    plot_performance_curve(ax, 'baseline', results['bo_results_total']['baseline_regrets_mean'],
                           results['bo_results_total']['baseline_regrets_std'], time_list)
    plot_performance_curve(ax, 'hyperbo+', results['bo_results_total']['gamma_regrets_mean'],
                           results['bo_results_total']['gamma_regrets_std'], time_list)
    ax.legend()
    fig.savefig(os.path.join(dir_path, 'regret_vs_iteration.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    results = np.load('results/test_hyperbo_plus_2022-09-11_08-39-05/results.npy', allow_pickle=True).item()
    # plot_estimated_prior(results)
    plot_hyperbo_plus(results)

