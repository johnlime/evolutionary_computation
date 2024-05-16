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

np.random.seed(0)

experiment_settings = {
    "novelty_selection": [False, True, True],
    "novelty_selection_prop": [0, 0.1, 0.9],
    "num": 3
}

for num in range(experiment_settings["num"]):
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
            novelty_selection = experiment_settings["novelty_selection"][num],
            novelty_k = novelty_k,
            novelty_selection_prop = \
                experiment_settings["novelty_selection_prop"][num],
            max_archive_length = max_archive_length,
            return_details = False)
        fitness_records[i] = f
        diversity_records[i] = d
        solution_records.append(s)

    tag = "NK: " + str(n) + ", " + str(k) + "; " + \
        "Novelty K: " + str(novelty_k) + \
        "; " + "Novelty Proportions: " + \
        str(experiment_settings["novelty_selection_prop"][num])
    experiment_results[tag] = deepcopy(fitness_records)
    # novelty_solutions_results[tag] = deepcopy(solution_records)
    diversity_results[tag] = deepcopy(diversity_records)

os.makedirs("assignment_results", exist_ok = True)

for novelty_selection_prop in experiment_settings["novelty_selection_prop"]:

    tag = "NK: " + str(n) + ", " + str(k) + "; " + \
        "Novelty K: " + str(novelty_k) + \
        "; " + "Novelty Proportions: " + \
        str(novelty_selection_prop)

    with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
              "_novelty_k_" + str(novelty_k) + \
              "_novelty_prop_" + str(novelty_selection_prop) + \
              "_experiment_results.pkl", 'wb') as filehandler:
        pickle.dump(experiment_results[tag], filehandler)

    # too much storage
    # with open("assignment_results/nk_" + str(n) + "_" + str(k) + "_novelty_k_" + str(novelty_k) + \
    #           "_novelty_prop_" + str(novelty_selection_prop) + "_solutions_results.pkl", 'wb') as filehandler:
    #     pickle.dump(novelty_solutions_results[tag], filehandler)

    with open("assignment_results/nk_" + str(n) + "_" + str(k) + \
              "_novelty_k_" + str(novelty_k) + \
              "_novelty_prop_" + str(novelty_selection_prop) + \
              "_diversity_results.pkl", 'wb') as filehandler:
        pickle.dump(diversity_results[tag], filehandler)


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
