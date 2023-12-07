import torch
import logging  # as log
from ambiegenvae.problems.abstract_problem import AbstractVAEProblem
from ambiegenvae.executors.abstract_executor import AbstractExecutor
import torch.nn as nn

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LKASVAEProblem(AbstractVAEProblem):
    """
    Represents a problem for the LKAS VAE.

    Args:
        executor (AbstractExecutor): The executor used for executing tests.
        transform (object): The transformation object.
        vae (nn.Module): The VAE model.
        n_var (int, optional): The number of decision variables. Defaults to 10.
        n_obj (int, optional): The number of objectives. Defaults to 1.
        n_ieq_constr (int, optional): The number of inequality constraints. Defaults to 1.
        min_fitness (float, optional): The minimum fitness value. Defaults to 0.95.
    """

    def __init__(
        self,
        executor: AbstractExecutor,
        transform: object,
        vae: nn.Module,
        n_var: int = 10,
        n_obj=1,
        n_ieq_constr=1,
        min_fitness=0.95,
    ):
        super().__init__(executor, vae, n_var, n_obj, n_ieq_constr)
        self.transform = transform
        self.min_fitness = min_fitness
        self._name = "LKASVAEProblem"

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the fitness of the given input.

        Args:
            x: The input to evaluate.
            out: The output dictionary to store the results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        fitness = self.executor.execute_test(x)
        log.info(f"Fitness output: {fitness}")
        out["F"] = fitness
        out["G"] = self.min_fitness - fitness * (-1)
