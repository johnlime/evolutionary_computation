import numpy as np

def update_archive(solution_archive, individual, max_archive_length):
    """
    solution_archive is only stored for calculating novelty before novelty selection
    The size of solution_archive should not matter to num_parent or num_children
    """

    # solution archive should be ordered from highest to lowest novelty
    if len(solution_archive) != 0:
        for i, solution in enumerate(solution_archive):
            if individual.novelty > solution.novelty:
                solution_archive.remove(solution)
                solution_archive.insert(i, individual)

    else:
        solution_archive.append(individual)


def get_diversity(gene_pool,
                  num_genes,
                  genome_length):
    # get standard deviation (axis = 0 is rows)
    diversity = np.empty((num_genes, genome_length))
    for i in range(num_genes):
        diversity[i] = gene_pool[i].genome
    diversity = np.std(diversity, axis = 1)
    # get average of all of the individual std genes
    diversity = np.mean(diversity)
    return diversity

def get_novelty(solution_archive, individual, k):
    if len(solution_archive) == 0:
        return 0

    distance_per_solution = np.empty(len(solution_archive))
    for i, solution in enumerate(solution_archive):
        distance_per_solution[i] = np.linalg.norm(np.array(solution.genome) - np.array(individual.genome),
                                                  ord = 2)

    if len(solution_archive) >= k:
        k_nearest_indices = np.argsort(distance_per_solution)[:k]
    else:
        k_nearest_indices = np.argsort(distance_per_solution)
    return np.mean(distance_per_solution[k_nearest_indices])
