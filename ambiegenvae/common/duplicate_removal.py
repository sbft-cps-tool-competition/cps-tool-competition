from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np

import logging 
from ambiegenvae.generators.abstract_generator import AbstractGenerator
log = logging.getLogger(__name__)


class AbstractDuplicateElimination(ElementwiseDuplicateElimination):
    """
    AbstractDuplicateElimination is a class that represents the abstract implementation of duplicate elimination.

    Attributes:
        generator (AbstractGenerator): The generator used for comparison.
        threshold (float): The threshold value for determining duplicates.

    Methods:
        is_equal(a, b): Checks if two elements are equal based on their vectors.
    """

    def __init__(self, generator:AbstractGenerator, threshold:float = 0.15):
        super().__init__()
        self.generator = generator
        self.threshold = threshold

    def is_equal(self, a, b):
        """
        Checks if two elements are equal based on their vectors.

        Args:
            a: The first element to compare.
            b: The second element to compare.

        Returns:
            bool: True if the elements are considered duplicates, False otherwise.
        """
        vector1 = np.array(a.X)
        vector2 = np.array(b.X)
        difference = self.generator.cmp_func(vector1, vector2)
        #if difference < self.threshold:
        #    log.info(f"Duplicate detected: {difference}")
        return difference < self.threshold

