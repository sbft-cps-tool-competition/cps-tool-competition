import os
import abc
import logging  # as log
from abc import ABC
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.generators.abstract_generator import AbstractGenerator
from typing import Tuple

log = logging.getLogger(__name__)


class AbstractExecutor(ABC):
    """
    Class for evaluating the fitness of the test scenarios.

    Attributes:
        generator (AbstractGenerator): The generator used to convert genotypes to phenotypes.
        test_validator (AbstractValidator, optional): The validator used to validate test scenarios.
        results_path (str, optional): The path to store the results.

    Methods:
        __init__: Initializes the AbstractExecutor object.
        execute_test: Executes a test and returns the fitness score and information about the test execution.
        _execute: Abstract method to be implemented by subclasses for executing a test.
        name: Returns the name of the executor.

    """

    def __init__(
        self,
        generator: AbstractGenerator,
        test_validator: AbstractValidator = None,
        results_path: str = None,
    ):
        """
        Initializes the AbstractExecutor object.

        Args:
            generator (AbstractGenerator): The generator used to convert genotypes to phenotypes.
            test_validator (AbstractValidator, optional): The validator used to validate test scenarios.
            results_path (str, optional): The path to store the results.

        """
        self.results_path = results_path
        self.test_validator = test_validator
        self.test_dict = {}
        self.generator = generator
        self._name = "AbstractExecutor"

        if results_path:
            os.makedirs(results_path, exist_ok=True)

        self.exec_counter = -1  # counts how many executions have been

    def execute_test(self, test) -> Tuple[float, str]:
        """
        Executes a test and returns the fitness score and information about the test execution.

        Args:
            test: The test case to be executed.

        Returns:
            Tuple[float, str]: A tuple containing the fitness score and information about the test execution.

        """
        self.exec_counter += 1  # counts how many executions have been
        self.test_dict[self.exec_counter] = {
            "test": test,
            "fitness": None,
            "info": None,
        }
        fitness = 0

        if self.test_validator:
            test = self.generator.genotype2phenotype(test)
            valid, info = self.test_validator.is_valid(test)
            log.info(f"Test validity: {valid}")
            log.info(f"Test info: {info}")
            if not valid:
                self.test_dict[self.exec_counter]["fitness"] = fitness
                self.test_dict[self.exec_counter]["info"] = info
                return float(fitness)
        try:
            fitness = self._execute(test)
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info

        except Exception as e:
            log.info(f"Error during execution {e}")
            self.test_dict[self.exec_counter]["info"] = "Error during execution of test"

        return float(fitness)

    @abc.abstractmethod
    def _execute(self, test) -> float:
        """
        Abstract method to be implemented by subclasses for executing a test.

        Args:
            test: The test case to be executed.

        Returns:
            float: The fitness score of the test execution.

        """
        pass

    @property
    def name(self) -> int:
        """
        Returns the name of the executor.

        Returns:
            int: The name of the executor.

        """
        return self._name
