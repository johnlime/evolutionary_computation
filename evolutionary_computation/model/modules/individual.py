import numpy as np

class Individual:
    def __init__(self, fitness_function, bit_string_length, continuous = False):
        if not continuous:
            self.genome = np.random.randint(
                low = 0, high = 2,
                size = bit_string_length)
        else:
            self.genome = np.random.rand(bit_string_length)
        self.fitness_function = fitness_function
        self.fitness = 0
        self.novelty = 0
        self.generation = 0

    def eval_fitness(self):
        self.fitness = self.fitness_function(self.genome)
        return self.fitness

    def assign_novelty(self, novelty):
        self.novelty = novelty
