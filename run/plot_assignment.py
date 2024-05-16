from evolutionary_computation import *
import pickle

bit_string_length = 15
n = bit_string_length
k = bit_string_length - 1
novelty_k = 5

experiment_settings = {
    "novelty_selection": [False, True, True],
    "novelty_selection_prop": [0, 0.1, 0.9],
    "num": 3
}

experiment_results = {}
diversity_results = {}

for novelty_selection_prop in experiment_settings["novelty_selection_prop"]:
    tag = "NK: " + str(n) + ", " + str(k) + "; " + \
        "Novelty K: " + str(novelty_k) + \
        "; " + "Novelty Proportions: " + \
        str(novelty_selection_prop)

    with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
              "_novelty_k_" + str(novelty_k) + \
              "_novelty_prop_" + str(novelty_selection_prop) + \
              "_experiment_results.pkl", 'rb') as filehandler:
        experiment_results[tag] = pickle.load(filehandler)

    with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
              "_novelty_k_" + str(novelty_k) + \
              "_novelty_prop_" + str(novelty_selection_prop) + \
              "_diversity_results.pkl", 'rb') as filehandler:
        diversity_results[tag] = pickle.load(filehandler)



# plotting
plot_mean_and_bootstrapped_ci_over_time(experiment_results,
                                        figure_path = "assignment_results/nk_" + \
                                            str(n) + "_" + str(k) + \
                                            "_novelty_k_" + str(novelty_k) + \
                                            "_novelty_prop_" + str(novelty_selection_prop) + \
                                            "_experiment_results.png")
plot_mean_and_bootstrapped_ci_over_time(diversity_results,
                                        name = "Diversity Over Generations",
                                        y_label = "Diversity",
                                        figure_path = "assignment_results/nk_" + \
                                            str(n) + "_" + str(k) + \
                                            "_novelty_k_" + str(novelty_k) + \
                                            "_novelty_prop_" + str(novelty_selection_prop) + \
                                            "_diversity_results.png")
