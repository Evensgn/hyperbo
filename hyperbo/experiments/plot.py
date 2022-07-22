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
        kernel_results = results['kernel_results'][kernel[0]]
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
        fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}.pdf'.format(kernel[0])))
        plt.close(fig)

        # visualize bo
        visualize_bo_results = kernel_results['visualize_bo_results']
        n_visualize_grid_points = visualize_bo_results['n_visualize_grid_points']
        for i in range(results['budget']):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.set_title('BO iteration {}'.format(i + 1))
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
            mean_inferred = visualize_bo_results['posterior_list'][i]['mean_inferred'].squeeze()
            std_inferred = visualize_bo_results['posterior_list'][i]['std_inferred'].squeeze()
            line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                           mean_inferred[:n_visualize_grid_points], label='inferred')[0]
            ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                            mean_inferred[:n_visualize_grid_points] - std_inferred[:n_visualize_grid_points],
                            mean_inferred[:n_visualize_grid_points] + std_inferred[:n_visualize_grid_points],
                            alpha=0.2, color=line.get_color())
            ax.legend()
            fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}_iteration_{}.pdf'.format(kernel[0], i)))
            plt.close(fig)


if __name__ == '__main__':
    pass

