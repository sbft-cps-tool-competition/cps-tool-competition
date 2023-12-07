import numpy as np
from ambiegenvae.generators.abstract_generator import AbstractGenerator
from ambiegenvae.sampling.abstract_sampling import AbstractSampling
from ambiegenvae.executors.abstract_executor import AbstractExecutor


class GreedySampling(AbstractSampling):
    """
    A class representing the greedy sampling strategy.

    This strategy selects the best phenotype out of a given number of randomly generated tests.

    Args:
        generator (AbstractGenerator): The generator used to generate random tests.
        greedy_executor (AbstractExecutor): The executor used to execute the tests.

    Attributes:
        executor (AbstractExecutor): The executor used to execute the tests.

    """

    def __init__(self, generator:AbstractGenerator, greedy_executor:AbstractExecutor) -> None:
        super().__init__(generator)
        self.executor = greedy_executor


    def _do(self, problem, n_samples, **kwargs):
        """
        Perform the greedy sampling strategy.

        Args:
            problem: The problem to be solved.
            n_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: An array of generated samples.

        """
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test = self._select_best_of_k(10)
            X[i] = test
            i += 1

        return X
    
    def _select_best_of_k(self, k:int):
        """
        Select the best phenotype out of k randomly generated tests.

        Args:
            k (int): The number of tests to generate.

        Returns:
            list: The best phenotype.

        """
        best_fitness = 0
        best_phenotype = []

        for i in range(k):
            test, valid = self.generator.generate_random_test()
            phenotype = self.generator.genotype

            fitness = self.executor.execute_test(test)
            if fitness < best_fitness:
                best_fitness = fitness
                best_phenotype = phenotype

        return best_phenotype
