# Evolutionary Computation

Implementation of evolutionary algorithm for discrete binary genomes as seen in CS/CSYS 352: Evolutionary Computation Fall 2022 at University of Vermont instructed by Prof. Nick Cheney.

## Features
- Elitism
- Random restart
- 2-point list crossover
- Truncation selection
- Tournament selection
- Scatter search (novelty search, diversity)
- N-K Fitness Landscape (implemented by Prof. Nick Cheney)

## Dependencies
- Numpy
- Matplotlib
- Scikits.bootstrap
- Scipy

## Get Started
Type the following in the command line.
```
export PYTHONPATH=$PWD
```

Run the evolutionary algorithm on N-K landscape.
```
python run/run_assignment.py
```

Save PNG's of fitness and diversity graphs for the search.
```
python run/plot_assignment.py
```

## Reference
- Sean, Luke (George Mason University). 2010. Essentials of Metaheuristics: A Set of Undergraduate Lecture Notes. Optimization.
