import numpy as np
import copy

"""
Gene Alteration Modules
"""

def crossover_module(children, parents,
                     crossover_ratio,
                     num_parents, num_children,
                     gen, indiv_count):

    crossover_count = int(num_children * crossover_ratio * 0.5)

    for i in range(crossover_count):
        # choose 2 different parent indices
        a, b = -1, -1
        while a == b:
            a, b = np.random.randint(low = 0,
                                     high = num_parents,
                                     size = 2)

        # choose 2 different gene indices
        c, d = -1, -1
        while c == d:
            c, d = np.random.randint(low = 1,
                                     high = parents[0].genome.size + 1,
                                     size = 2)

        # ensure that c is smaller than d
        if c > d:
            tmp = copy.copy(c)
            c = copy.copy(d)
            d = tmp # tmp is already a copied version of c

        # crossover process
        child_a = copy.deepcopy(parents[a])
        child_b = copy.deepcopy(parents[b])
        child_a.genome[c:d] = parents[b].genome[c:d]
        child_b.genome[c:d] = parents[a].genome[c:d]
        children.append(child_a)
        children.append(child_b)

        # crossed-over children: record generation that they were created for tracking
        children[indiv_count].generation = gen
        children[indiv_count + 1].generation = gen
        indiv_count += 2

    assert len(children) == num_parents + crossover_count * 2

    # how unfortunate. indiv_count has become a mutable type at some point
    assert indiv_count == num_parents + crossover_count * 2

    return crossover_count, indiv_count


def mutation_module(children, parents,
                    crossover_count, crossover_mutation_ratio,
                    num_parents, num_children,
                    gen, indiv_count,
                    num_elements_to_mutate = 1,
                    continuous = False,
                    mutation_size = 1.0):

    """
    children = [parents, crossover(_mutation_parents)]
    """

    crossover_mutation_count = int(crossover_count * 2 * crossover_mutation_ratio)

    # the parents of crossover x mutation are the crossover genes
    crossover_mutation_parents = children[num_parents:] # crossover mutation should replace crossover (no deepcopy)
    assert len(crossover_mutation_parents) == crossover_count * 2

    # Since crossover_mutation_ratio is a thiing, assertion of
    # len(crossover_mutation_parents) == crossover_mutation_count doesn't makes sense unless
    # crossover_mutation_ratio = 1.0
    assert indiv_count == num_parents + crossover_count * 2


    for i in range(num_children - crossover_count * 2 + crossover_mutation_count): # crossover mutation + mutation only
        child = None

        # crossover x mutation (up until crossover_mutation_count)
        if i < crossover_mutation_count:
            index = np.random.randint(low = 0,
                                      high = crossover_count * 2,
                                      size = 1)[0]
            # crossover mutation should replace crossover (which comes after the parents)
            child = crossover_mutation_parents[index]

            # are the pointers working?
            assert children[num_parents + index] == child

        # parents x mutation (child variable is filled here)
        else: # choose a parent index
            index = np.random.randint(low = 0,
                                      high = num_parents,
                                      size = 1)[0]
            child = copy.deepcopy(parents[index])
            children.append(child)

            # mutated children generation tracking
            assert children[indiv_count] == child # are the pointers working?
            children[indiv_count].generation = gen
            indiv_count += 1

        # the actual mutation process per genome
        if continuous:
            child.genome = child.genome.astype(float)
            child.genome += (np.random.rand(child.genome.size) * 2 - 1) * mutation_size
        else:
            used_indices = []
            # only mutate one of the genome
            for j in range(num_elements_to_mutate):
                sample_index = np.random.randint(low = 0,
                                                 high = child.genome.size,
                                                 size = 1)[0]
                if sample_index in used_indices:
                    j -= 1
                else:
                    child.genome[sample_index] = \
                        np.absolute(child.genome[sample_index] - 1)
                    used_indices.append(sample_index)

    assert len(children) == num_parents + num_children

    return crossover_mutation_count, indiv_count


"""
Selection Modules
"""

def truncation_selection_module(parents, children,
                                downhill_prob,
                                num_parents, num_children,
                                novelty_search = False,
                                novelty_selection_prob = 0):
    # selection procedure
    # selection is conducted within the children (to-be-parent) gene pool

    # order index for best children score
    children_fitness_score = np.zeros(num_parents + num_children)
    children_novelty_score = np.zeros(num_parents + num_children)

    for i in range(num_parents + num_children):
        children_fitness_score[i] = copy.deepcopy(children[i].fitness)
        children_novelty_score[i] = copy.deepcopy(children[i].novelty) # useless if novelty_search is off; reduce if statements

    children_fitness_order_indices = np.flip(np.argsort(children_fitness_score))
    children_novelty_order_indices = np.flip(np.argsort(children_novelty_score))

    # fitness and novelty proportion
    if not novelty_search:
        novelty_selection_prob = 0

    # index selection memory
    used_fitness_indices = [-1]
    fitness_bad_index = -1
    used_novelty_indices = [-1]
    novelty_bad_index = -1

    # references
    children_general_order_indices = None
    used_indices = None
    for i in range(num_parents):
        # selection of novelty order indices or fitness order indices
        if np.random.uniform(low=0.0, high=1.0) > novelty_selection_prob:
            children_general_order_indices = children_fitness_order_indices
            used_indices = used_fitness_indices
            bad_index = fitness_bad_index
        else:
            children_general_order_indices = children_novelty_order_indices
            used_indices = used_novelty_indices
            bad_index = novelty_bad_index

        # stochastic selection of better answer
        if np.random.uniform(low=0.0, high=1.0) > downhill_prob:
            selected_index = copy.deepcopy(
                children_general_order_indices[i])
            parents.append(copy.deepcopy(children[selected_index]))
            used_indices.append(selected_index)

        # stochastic selection of bad answer
        else:
            while bad_index in used_indices:
                bad_index = np.random.randint(num_parents,
                                              num_parents + num_children,
                                              size = 1)[0]
            selected_index = copy.deepcopy(
                children_general_order_indices[bad_index])
            parents.append(copy.deepcopy(children[selected_index]))
            used_indices.append(selected_index)



def tournament_selection_module(parents, children,
                                tournament_size, num_tournament_winners,
                                num_parents, num_children,
                                novelty_search = False):
    # tournament selection with replacement
    # parameters
    tournament_size = 4
    num_tournament_winners = 2
    # num_parents, num_children

    for _ in range(int(np.ceil(num_parents / num_tournament_winners))):
        tournament_indices = np.random.randint(
            low = 0,
            high = num_parents + num_children,
            size = tournament_size)

        # just pick out the ones in the tournament
        tournament_scores = np.empty(tournament_size)
        for i in range(tournament_size):
            # tournament_scores is in the same order as tournament_indices
            if novelty_search:
                tournament_scores[i] = copy.deepcopy(children[tournament_indices[i]].novelty)
            else:
                tournament_scores[i] = copy.deepcopy(children[tournament_indices[i]].fitness)

        # flip for highest to lowest
        tournament_winners = np.flip(np.argsort(tournament_scores))[:num_tournament_winners]
        for i in tournament_winners:
            parents.append(copy.deepcopy(children[tournament_indices[i]]))
    if len(parents) > num_parents:
        parents = parents[:num_parents]
