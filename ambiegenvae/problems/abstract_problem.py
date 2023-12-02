from abc import ABC, abstractmethod
from pymoo.core.problem import ElementwiseProblem
from ambiegenvae.executors.abstract_executor import AbstractExecutor
import numpy as np
import torch.nn as nn


class AbstractVAEProblem(ElementwiseProblem, ABC):
    """
    This is the base class for performing solution evaluation using a Variational Autoencoder (VAE).

    Attributes:
        executor (AbstractExecutor): The executor used for evaluating solutions.
        vae (nn.Module): The VAE model used for encoding and decoding solutions.
        n_var (int): The number of decision variables.
        n_obj (int): The number of objectives.
        n_ieq_constr (int): The number of inequality constraints.
    """

    def __init__(
        self,
        executor: AbstractExecutor,
        vae: nn.Module,
        n_var: int = 10,
        n_obj=1,
        n_ieq_constr=1,
    ):
        x_u = np.ones(n_var) * 3
        x_l = np.ones(n_var) * (-3)
        self.executor = executor
        self.vae = vae
        self._name = "AbstractVAEProblem"

        super().__init__(
            n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=x_l, xu=x_u
        )

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Abstract method for evaluating a solution.

        Args:
            x: The solution to be evaluated.
            out: The output array to store the evaluation results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    @property
    def name(self) -> int:
        """
        Get the name of the problem.

        Returns:
            str: The name of the problem.
        """
        return self._name

    # You can add non-abstract methods here if needed


class AbstractProblem(ElementwiseProblem, ABC):
    """
    This is the base class for performing solution evaluation.

    Attributes:
        executor (AbstractExecutor): The executor used for evaluating solutions.
        n_var (int): The number of decision variables.
        n_obj (int): The number of objectives.
        n_ieq_constr (int): The number of inequality constraints.
        _name (str): The name of the problem.
    """

    def __init__(
        self, executor: AbstractExecutor, n_var: int = 10, n_obj=1, n_ieq_constr=1
    ):
        self.executor = executor
        self._name = "AbstractProblem"

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr)

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Abstract method for evaluating a solution.

        Args:
            x: The solution to be evaluated.
            out: The output array to store the evaluated objectives and constraints.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    @property
    def name(self) -> int:
        """
        Get the name of the problem.

        Returns:
            str: The name of the problem.
        """
        return self.name
