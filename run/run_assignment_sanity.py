import numpy as np
from evolutionary_computation import *
import pickle
import os
from copy import deepcopy

experiment_results = {}
#solutions_results = {}
diversity_results = {}

num_runs = 20
total_generations = 100
num_elements_to_mutate = 1
bit_string_length = 15
num_parents = 20
num_children = 20

novelty_k = 5
max_archive_length = 100

n = bit_string_length
k = bit_string_length-1
fitness_landscape = Landscape(n, k)

np.random.seed(50)

fitness_records = np.empty((num_runs, total_generations))
diversity_records = np.empty((num_runs, total_generations))
solution_records = []

for i in range(num_runs):
    print("Run " + str(i))
    f, s, d = evolutionary_algorithm(
        fitness_function = fitness_landscape.get_fitness,
        total_generations = total_generations,
        num_parents = num_parents,
        num_children = num_children,
        continuous = False,
        genome_length = bit_string_length,
        num_elements_to_mutate = num_elements_to_mutate,
        crossover = False,
        restart_every = 0,
        downhill_prob = 0.01,
        tournament_selection = False,
        novelty_selection = False,
        novelty_k = novelty_k,
        novelty_selection_prop = 0,
        max_archive_length = max_archive_length,
        return_details = False)
    fitness_records[i] = f
    diversity_records[i] = d
    solution_records.append(s)

tag = "nk_" + str(n) + "_" + str(k)

experiment_results[tag] = deepcopy(fitness_records)
# novelty_solutions_results[tag] = deepcopy(solution_records)
diversity_results[tag] = deepcopy(diversity_records)

os.makedirs("assignment_results", exist_ok = True)

with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
          "_experiment_results.pkl", 'wb') as filehandler:
    pickle.dump(experiment_results, filehandler)

with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
          "_diversity_results.pkl", 'wb') as filehandler:
    pickle.dump(diversity_results, filehandler)


experiment_results = {}
diversity_results = {}

with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
          "_experiment_results.pkl", 'rb') as filehandler:
    experiment_results = pickle.load(filehandler)

with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
          "_diversity_results.pkl", 'rb') as filehandler:
    diversity_results = pickle.load(filehandler)

plot_mean_and_bootstrapped_ci_over_time(experiment_results,
                                        figure_path = "assignment_results/nk_" + \
                                            str(n) + "_" + str(k) + \
                                            "_experiment_results.png")
plot_mean_and_bootstrapped_ci_over_time(diversity_results,
                                        name = "Diversity Over Generations",
                                        y_label = "Diversity",
                                        figure_path = "assignment_results/nk_" + \
                                            str(n) + "_" + str(k) + \
                                            "_diversity_results.png"
                                        )
