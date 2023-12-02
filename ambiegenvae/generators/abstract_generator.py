import abc
import typing
import numpy as np


class AbstractGenerator(abc.ABC):
    """Abstract class for all generators."""

    def __init__(self, solution_size: int):
        """Initialize the generator.

        Args:
            solution_size (int): Size of the solution.
        """
        self.size = solution_size
        self._name = "AbstractGenerator"

    @property
    @abc.abstractmethod
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        pass

    @property
    def name(self) -> int:
        """Name of the generator.

        Returns:
            int: Name of the generator.
        """
        return self._name

    @property
    @abc.abstractmethod
    def genotype(self) -> typing.List[float]:
        """Genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        pass

    @abc.abstractmethod
    def cmp_func(self, x: np.ndarray, y: np.ndarray) -> float:
        """Comparison function for the generator.

        Args:
            x (np.ndarray): First input array.
            y (np.ndarray): Second input array.

        Returns:
            float: Difference of two vectors.
        """
        pass

    @abc.abstractmethod
    def genotype2phenotype(
        self, genotype: typing.List[float]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Convert genotype to phenotype.

        Args:
            genotype (list): Genotype of the generator.

        Returns:
            tuple: Phenotype of the generator.
        """
        pass

    @abc.abstractmethod
    def set_genotype(self, phenotype):
        """Set the phenotype of the generator.

        Args:
            phenotype (list): Phenotype of the generator.
        """
        pass

    @abc.abstractmethod
    def get_phenotype(self):
        """Get the phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        pass

    @abc.abstractmethod
    def generate_random_test(self) -> (typing.List[float], bool):
        """Generate random test samples.

        Returns:
            tuple: Generated samples and a boolean value.
        """
        pass

    @abc.abstractmethod
    def visualize_test(self, test: typing.List[float], save_path: str = None):
        """Visualize a test.

        Args:
            test (list): Test to visualize.
            save_path (str, optional): Path to save the visualization. Defaults to None.
        """
        pass
