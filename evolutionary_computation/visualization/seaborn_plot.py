import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)

import scipy.stats # for finding statistical significance

def plot_mean_and_bootstrapped_ci_over_time(input_data = None,
                                            n_samples = 20000,
                                            name = "Fitness Over Generations",
                                            x_label = "Generations",
                                            y_label = "Fitness",
                                            fig_size_change = False,
                                            figure_path = None):
    """
    parameters:
    input_data: (numpy array of shape {dict_key: (max_k, max_gen)}) solution metric to plot
    name: (string) name for legend
    x_label: (string) x axis label
    y_label: (string) y axis label

    returns:
    None
    """
    if fig_size_change:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

    for key in input_data:
        recorded_matrix = input_data[key]
        max_k = recorded_matrix.shape[0]
        max_gen = recorded_matrix.shape[1]
        confidence_intervals = np.empty((max_gen, 2))
        # confidence interval
        for gen in range(max_gen):
            confidence_intervals[gen] = np.array(bootstrap.ci(
                data = recorded_matrix[:, gen],
                statfunction = np.mean,
                n_samples = n_samples))
        plt.plot(np.mean(recorded_matrix, axis = 0), label = key)
        plt.fill_between(np.arange(max_gen),
                         confidence_intervals[:, 0],
                         confidence_intervals[:, 1],
                         alpha = 0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if figure_path == None:
        plt.show()
        plt.close()

    else:
        plt.savefig(figure_path)
        plt.close()
