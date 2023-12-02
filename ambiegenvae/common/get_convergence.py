"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for getting the convergence of the search algoritm (best values at each generation)
"""

import numpy as np


def get_convergence(res):
    """
    It takes the result of the genetic algorithm and returns a list of the best fitness values of each generation.

    Args:
      res: the result of the genetic algorithm
    """
    res_dict = {}
    generations = np.arange(0, len(res.history), 1)
    convergence = []
    algorithm = res.algorithm
    pop_size = algorithm.pop_size
    n_offsprings = algorithm.n_offsprings

    for gen in generations:
        population = res.history[gen].pop.get("F")*(-1)
        best = np.max(population)
        convergence.append(float(best))

    step = n_offsprings
    evaluations = np.arange(
        pop_size, len(res.history) * n_offsprings + pop_size, step
    )

    for i in range(len(evaluations)):
        res_dict[str(evaluations[i])] = convergence[i]
    return res_dict
