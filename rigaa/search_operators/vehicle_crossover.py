import random as rm
import logging as log
import numpy as np
from pymoo.core.crossover import Crossover

from rigaa.solutions import VehicleSolution

# this is the crossover operator for the vehicle problem
class VehicleCrossover(Crossover):
    """
    Module to perform the crossover
    """

    def __init__(self, cross_rate, map_size, executor):
        super().__init__(2, 2)
        self.cross_rate = cross_rate
        self.map_size = map_size
        self.executor = executor

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        Y = np.full_like(X, None, dtype=object)
        # for each mating provided
        for k in range(n_matings):

            r = np.random.random()
            s_a, s_b = X[0, k, 0], X[1, k, 0]
            if r < self.cross_rate:
                log.debug("Crossover performed on individuals %s and %s", s_a, s_b)
                tc_a = s_a.states.copy()
                tc_b = s_b.states.copy()

                min_len = min(len(tc_a), len(tc_b))
                crossover_point = rm.randint(1, min_len - 1)
                log.debug("Crossover point: %d", crossover_point)

                if len(s_a.states) > 2 and len(s_b.states) > 2:

                    offa = VehicleSolution(self.map_size, self.executor)
                    offb = VehicleSolution(self.map_size, self.executor)
                    # one point crossover
                    offa.states[:crossover_point] = tc_a[:crossover_point]
                    offa.states[crossover_point:] = tc_b[crossover_point:]
                    offb.states[:crossover_point] = tc_b[:crossover_point]
                    offb.states[crossover_point:] = tc_a[crossover_point:]
                    
                    Y[0, k, 0], Y[1, k, 0] = offa, offb

                else:
                    log.debug("Not enough states to perform crossover")
                    Y[0, k, 0], Y[1, k, 0] = s_a, s_b
            else:
                Y[0, k, 0], Y[1, k, 0] = s_a, s_b

        return Y

