"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-11-02
Description: script for evaluating the diversity of test scenarios
"""

import numpy as np
from ambiegenvae.generators.abstract_generator import AbstractGenerator

def calc_novelty(x_1:np.ndarray, x_2:np.ndarray, generator: AbstractGenerator) -> float:
    """
    > The function takes two states and a problem type as input and returns the novelty of the two
    states

    Args:
      state1: the first state to compare
      state2: the state to compare to

    Returns:
      The novelty of the solution relative to the other solutions in the test suite.
    """

    novelty = generator.cmp_func(x_1, x_2)
    return novelty
