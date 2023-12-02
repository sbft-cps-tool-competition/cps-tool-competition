import numpy as np
from ambiegenvae.crossover.abstract_crossover import AbstractCrossover



class OnePointCrossover(AbstractCrossover):
    def __init__(self, cross_rate: float = 0.9):
        """
        Initialize the OnePointCrossover class.

        Parameters:
        - cross_rate (float): The crossover rate, which determines the probability of crossover occurring.

        Returns:
        None
        """
        super().__init__(cross_rate)


    def _do_crossover(self, problem, a, b) -> tuple:
        """
        Perform one-point crossover on two parent individuals.

        Parameters:
        - problem: The problem instance.
        - a: The first parent individual.
        - b: The second parent individual.

        Returns:
        - tuple: A tuple containing the two offspring individuals.
        """
        n_var = problem.n_var
        n = np.random.randint(1, n_var)
        off_a = np.concatenate([a[:n], b[n:]])
        off_b = np.concatenate([b[:n], a[n:]])
        return off_a, off_b