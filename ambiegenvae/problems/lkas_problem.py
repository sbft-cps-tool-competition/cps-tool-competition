from ambiegenvae.problems.abstract_problem import AbstractProblem
import logging  # as log
from ambiegenvae.executors.abstract_executor import AbstractExecutor
import time

log = logging.getLogger(__name__)


class LKASProblem(AbstractProblem):
    """
    Represents the LKAS (Lane Keeping Assist System) problem.

    This class inherits from the AbstractProblem class and implements the necessary methods
    for evaluating the fitness of a given solution.

    Parameters:
    - executor: An instance of the AbstractExecutor class used for executing the test.
    - n_var: The number of variables in the solution (default: 10).
    - n_obj: The number of objectives (default: 1).
    - n_ieq_constr: The number of inequality constraints (default: 1).
    - min_fitness: The minimum fitness value (default: 0.95).
    """

    def __init__(
        self,
        executor: AbstractExecutor,
        n_var: int = 10,
        n_obj=1,
        n_ieq_constr=1,
        min_fitness=0.95,
    ):
        super().__init__(executor, n_var, n_obj, n_ieq_constr)
        self.min_fitness = min_fitness
        self._name = "LKASProblem"

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the fitness of a given solution.

        Parameters:
        - x: The solution to be evaluated.
        - out: A dictionary to store the fitness and constraint values.
        - *args, **kwargs: Additional arguments and keyword arguments.

        Returns:
        None
        """
        test = x
        start = time.time()
        fitness = self.executor.execute_test(test)
        log.info(f"Time to evaluate: {time.time() - start}")
        log.info(f"Fitness output: {fitness}")
        out["F"] = fitness
        out["G"] = self.min_fitness - fitness * (-1)
