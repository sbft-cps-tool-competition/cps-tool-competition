from itertools import combinations
import logging as log
from rigaa.utils.calc_novelty import calc_novelty
import config as cf

def get_stats(res, problem, algo):
    """
    It takes the results of the optimization and returns a dictionary with the fitness, novelty, and
    convergence of the optimization

    Args:
      res: the result of the optimization
      problem: the problem we're trying to solve

    Returns:
      A dictionary with the fitness, novelty, and convergence of the results.
    """
    res_dict = {}
    gen = len(res.history) - 1
    results = []
    population = -res.history[gen].pop.get("F")
    if algo != "nsga2" and algo != "rigaa":
        population = sorted(population, key=lambda x: x[0], reverse=True)
    for i in range(cf.ga["test_suite_size"]):


        #result = res.history[gen].pop.get("F")[i][0]
        results.append(population[i][0])

    gen = len(res.history) - 1
    novelty_list = []
    test_population = res.history[gen].pop.get("X")
    if algo != "nsga2" and algo != "rigaa":
        test_population = sorted(test_population, key=lambda x: abs(x[0].fitness), reverse=True)
    for i in combinations(range(0, cf.ga["test_suite_size"]), 2):
        current1 = test_population[i[0]] #res.history[gen].pop.get("X")[i[0]]
        current2 = test_population[i[1]] #res.history[gen].pop.get("X")[i[1]]
        nov = calc_novelty(current1[0].states, current2[0].states, problem)
        novelty_list.append(nov)
    novelty = sum(novelty_list) / len(novelty_list)

    log.info("The highest fitness found: %f", max(results))
    log.info("Average diversity: %f", novelty)
    res_dict["fitness"] = results
    res_dict["novelty"] = novelty

    return res_dict