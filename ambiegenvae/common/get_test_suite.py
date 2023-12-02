"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for exctacting the generated test scenarios
"""

import logging #as log
log = logging.getLogger(__name__)


def get_test_suite(res, kappas=True):
    """
    It takes the last generation of the population and returns a dictionary of 30 test cases

    Args:
      res: the result of the genetic algorithm

    Returns:
      A dictionary of 30 test cases.
    """
    test_suite = {}
    gen = len(res.history) - 1
    algorithm = res.algorithm
    pop_size = algorithm.pop_size
    generator = algorithm.problem.executor.generator

    population = res.history[gen].pop.get("X")
    for i in range(pop_size):
        result = population[i]
        if not(kappas):
            result = generator.genotype2phenotype(result)
            test_suite[str(i)] = result
        else:
            new_result = [float(x) for x in result]
            test_suite[str(i)] = new_result

    log.info("Test suite of %d test scenarios generated", pop_size)
    return test_suite
