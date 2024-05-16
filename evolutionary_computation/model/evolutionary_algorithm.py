import numpy as np
import copy
from evolutionary_computation.model.modules import *


def evolutionary_algorithm(fitness_function = None,
                           total_generations = 100,
                           num_parents = 10,
                           num_children = 10,
                           continuous = False,
                           genome_length = 10,
                           num_elements_to_mutate = 1,
                           mutation_size_start = 1.0,
                           mutation_size_end = 0.1,
                           crossover = False,
                           crossover_ratio = 0.6,
                           crossover_mutation_ratio = 0.25,
                           restart_every = 0,
                           downhill_prob = 0.2,
                           tournament_selection = False,
                           tournament_size = 4,
                           num_tournament_winners = 2,
                           novelty_selection = True,
                           novelty_k = 5,
                           novelty_selection_prop = 0,
                           max_archive_length = 100,
                           return_details = False):
    """
    Evolutinary Algorithm (copied from the basic hillclimber in our last assignment)

    parameters:
    fitness_funciton: (callable function) that return the fitness of a genome
                       given the genome as an input parameter (e.g. as defined in Landscape)
    total_generations: (int) number of total iterations for stopping condition
    num_parents: (int) the number of parents we downselect to at each generation (mu)
    num_children: (int) the number of children (note: parents not included in this count) that we baloon to each generation (lambda)
    genome_length: (int) length of the genome to be evoloved
    num_elements_to_mutate: (int) number of alleles to modify during mutation (0 = no mutation)
    mutation_size_start: (float) scaling parameter of the magnitidue of mutations for floating point vectors at the beginning of search
    mutation_size_end: (float) scaling parameter of the magnitidue of mutations for floating point vectors at the end of search (note: if same as mutation_size_start, mutation rate is static, otherwise mutation rate is linearly interpolated between the two)
    crossover: (bool) whether to perform crossover when generating children
    tournament_size: (int) number of individuals competing in each tournament
    num_tournament_winners: (int) number of individuals selected as future parents from each tournament (must be less than tournament_size)

    returns:
    fitness_over_time: (numpy array) track record of the top fitness value at each generation
    solutions_over_time: (numpy array) track record of the top genome value at each generation
    diversity_over_time: (numpy array) track record of the population genetic diversity at each generation
    """
    # initialize record keeping
    fitness_over_time = np.empty(total_generations)
    diversity_over_time = np.empty(total_generations)
    solutions_over_time = []

    # novelty distance archive
    solution_archive = []
    if not novelty_selection:
        max_archive_length = 0

    # the initialization proceedure
    parents = [] # num_parents
    for i in range(num_parents):
        parents.append(Individual(fitness_function, genome_length))


    children = [] # num_parents + num_children

    # only one best solution and score
    parent_best_solution = None
    parent_best_score = 0 # per generation
    parent_best_generation = 0 # per generation
    best_solution = None
    best_score = 0
    best_generation = 0

    for i in range(num_parents):
        # get population fitness
        parents[i].eval_fitness()

        # get population novelty
        novelty = get_novelty(solution_archive, parents[i], novelty_k)
        parents[i].assign_novelty(novelty)
        update_archive(solution_archive, parents[i], max_archive_length)

    for gen in range(total_generations): # repeat
        # the modification procedure
        # inheritance
        children = copy.deepcopy(parents) # "children" has num_parents for now

        # for children generation tracking
        indiv_count = copy.copy(num_parents)
        assert (len(children) == num_parents) and (len(parents) == num_parents)

        # number of children we need to generate is \lambda (+ \mu)
        # crossover
        if crossover != True:
            crossover_ratio = 0
            crossover_mutation_ratio = 0

        # mutation size change
        mutation_size = mutation_size_start + (mutation_size_end - mutation_size_start) * gen / (total_generations - 1)

        crossover_count, indiv_count = crossover_module(children, parents,
                                                        crossover_ratio,
                                                        num_parents, num_children,
                                                        gen, indiv_count)

        assert len(children) == num_parents + crossover_count * 2
        assert indiv_count == len(children)

        # mutation
        _, indiv_count = mutation_module(children, parents,
                                         crossover_count, crossover_mutation_ratio,
                                         num_parents, num_children,
                                         gen, indiv_count,
                                         num_elements_to_mutate,
                                         continuous = continuous)

        assert len(children) == num_parents + num_children
        assert indiv_count == len(children)

        # clear out parents before assessment and replacement
        parents = []

        # the assessement procedure
        # the children gene pool consists of \mu + \lambda
        for i in range(num_children):
            children[num_parents + i].eval_fitness()

            # set novelty
            novelty = get_novelty(solution_archive, children[num_parents + i], novelty_k)
            children[num_parents + i].assign_novelty(novelty)
            update_archive(solution_archive, children[num_parents + i], max_archive_length)
        # children_generation array should be filled at this point

        # diversity measurement
        diversity = get_diversity(children,
                                  num_parents + num_children,
                                  genome_length)

        # tournament selection
        if tournament_selection == False:
            truncation_selection_module(parents, children,
                                        downhill_prob,
                                        num_parents, num_children,
                                        novelty_selection,
                                        novelty_selection_prop)


        else:
            # tournament selection with replacement
            # fitness + novelty is NOT implemented here
            # if novelty_selection, novelty proportion is assumed to be 1.0

            tournament_selection_module(parents, children,
                                        tournament_size, num_tournament_winners,
                                        num_parents, num_children,
                                        novelty_selection)

        assert len(parents) == num_parents

        # Clear out children, because we're not using it anymore
        children = []

        # Track generation progress
        parents_score = np.zeros(num_parents)
        for i in range(num_parents):
            parents_score = parents[i].fitness
        best_index = np.flip(np.argsort(parents_score))[0]
        parent_best_score = parents[best_index].fitness
        parent_best_solution = parents[best_index].genome
        parent_best_generation = parents[best_index].generation

        # random restart
        if restart_every != 0:
            if (gen + 1) % restart_every == 0 or gen + 1 == total_generations: # random restart wins
                # save current parent if it's the best score (copy of procedure with no random restart
                if parent_best_score > best_score:
                    best_solution = copy.deepcopy(parent_best_solution)
                    best_score = copy.copy(parent_best_score)
                    best_generation = copy.copy(parent_best_generation)
                # initialize population
                parents = []
                for i in range(num_parents):
                    parents.append(Individual(fitness_function,
                                              genome_length))
                    # fitness only
                    parents[i].eval_fitness()

                    # novelty only
                    novelty = get_novelty(solution_archive, individual, novelty_k)
                    parents[i].assign_novelty(novelty)
                    update_archive(solution_archive, parents[i], max_archive_length)

        else:
            best_solution = copy.deepcopy(parent_best_solution)
            best_score = copy.copy(parent_best_score)
            best_generation = copy.copy(parent_best_generation)

        # record keeping
        fitness_over_time[gen] = copy.copy(best_score) # becomes novelty over time if novelty_selection
        solutions_over_time.append(copy.deepcopy(best_solution))
        diversity_over_time[gen] = copy.copy(diversity)

    if return_details:
        return best_solution, best_score, best_generation, fitness_over_time, solutions_over_time, diversity_over_time
    else:
        return fitness_over_time, solutions_over_time, diversity_over_time
