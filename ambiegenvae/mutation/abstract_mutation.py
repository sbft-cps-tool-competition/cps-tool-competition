from pymoo.core.mutation import Mutation
import numpy as np
import abc


class AbstractMutation(Mutation, abc.ABC):
    """
    Abstract base class for mutations in a genetic algorithm.

    Attributes:
        mut_rate (float): The mutation rate, indicating the probability of mutation for each individual.

    Methods:
        _do: Perform mutation on the given population.
        _do_mutation: Abstract method to be implemented by subclasses for performing mutation on an individual.

    """

    def __init__(self, mut_rate: float = 0.4):
        super().__init__()
        self.mut_rate = mut_rate

    def _do(self, problem, X, **kwargs):
        """
        Perform mutation on the given population.

        Args:
            problem: The problem instance.
            X (np.ndarray): The population to be mutated.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The mutated population.

        """
        for i in range(len(X)):
            r = np.random.random()
            if r < self.mut_rate:
                X[i] = self._do_mutation(X[i])
        return X

    @abc.abstractmethod
    def _do_mutation(self, x) -> np.ndarray:
        """
        Abstract method to be implemented by subclasses for performing mutation on an individual.

        Args:
            x: The individual to be mutated.

        Returns:
            np.ndarray: The mutated individual.

        """
        pass
