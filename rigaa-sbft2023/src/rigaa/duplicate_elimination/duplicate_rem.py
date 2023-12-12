import logging as log
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import time
# It's a duplicate elimination that compares the states of the two elements


class DuplicateElimination(ElementwiseDuplicateElimination):
    '''
    A class to eliminate duplicates in the population.
    '''

    def is_equal(self, a, b):
        state1 = a.X[0].states
        state2 = b.X[0].states

        # Calculating the novelty of the two states.
        novelty = abs(a.X[0].calculate_novelty(state1, state2))
        if novelty  < 0.5:
            log.debug("Duplicate %s and %s found", a.X[0], b.X[0])
        return novelty < 0.5 
