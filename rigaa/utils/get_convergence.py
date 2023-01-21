
import numpy as np
import config as cf
def get_convergence(res, n_offsprings):
    """
    It takes the result of the genetic algorithm and returns a list of the best fitness values of each generation.

    Args:
      res: the result of the genetic algorithm
    """
    res_dict = {}
    generations = np.arange(0, len(res.history), 1)
    convergence = []
    for gen in generations:
        population = -res.history[gen].pop.get("F")
        population = sorted(population, key=lambda x: x[0], reverse=True)
        convergence.append(population[0][0])

    step = n_offsprings
    evaluations = np.arange(cf.ga["pop_size"], len(res.history)*n_offsprings + cf.ga["pop_size"], step)

    for i in range(len(evaluations)):
        res_dict[str(evaluations[i])] = convergence[i]
    return res_dict