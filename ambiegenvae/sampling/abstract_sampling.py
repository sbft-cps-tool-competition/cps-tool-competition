from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from ambiegenvae.generators.abstract_generator import AbstractGenerator


class AbstractSampling(Sampling, ABC):
    """
    AbstractSampling is an abstract base class for sampling methods.

    It provides a common interface for different sampling strategies.

    Args:
        generator (AbstractGenerator): The generator used for sampling.

    Attributes:
        generator (AbstractGenerator): The generator used for sampling.

    """

    def __init__(self, generator:AbstractGenerator) -> None:
        super().__init__()
        self.generator = generator


    def _do(self, problem, n_samples, **kwargs):
        """
        Perform the sampling process.

        Args:
            problem: The problem to be sampled.
            n_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: An array of generated samples.

        """
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test, valid = self.generator.generate_random_test()
            if valid:
                test = self.generator.genotype
                X[i] = np.array(test)
                i += 1

        return X
