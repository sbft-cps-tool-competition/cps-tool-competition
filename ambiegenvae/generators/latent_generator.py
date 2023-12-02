
import typing
import numpy as np
from ambiegenvae.generators.abstract_generator import AbstractGenerator
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentGenerator(AbstractGenerator):
    """Abstract class for all generators."""

    def __init__(
        self,
        solution_size: int,
        mean: float = 0.0,
        std: float = 1.0,
        original_gen: AbstractGenerator = None,
        model: nn.Module = None,
        transform: object = None,
    ):
        """Initialize the generator.

        Args:
            solution_size (int): Size of the solution.
            mean (float, optional): Mean value for generating random tests. Defaults to 0.0.
            std (float, optional): Standard deviation for generating random tests. Defaults to 1.0.
            original_gen (AbstractGenerator, optional): Original generator. Defaults to None.
            model (nn.Module, optional): Model for decoding tests. Defaults to None.
            transform (object, optional): Transformation object for preprocessing tests. Defaults to None.
        """
        super().__init__(solution_size)
        self.vector = np.zeros(self.size)
        self.mean = mean
        self.std = std
        self.orig_gen = original_gen
        self.model = model
        self.transform = transform

    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size

    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.vector

    def cmp_func(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compare two tests.

        Args:
            x (np.ndarray): First test.
            y (np.ndarray): Second test.

        Returns:
            float: Difference between the two tests.
        """
        x = self.decode_test(x)
        y = self.decode_test(y)
        difference = self.orig_gen.cmp_func(x, y)
        return difference

    def decode_test(
        self, test: typing.List[float]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Decode a test.

        Args:
            test (np.array): Test to decode.

        Returns:
            tuple: Decoded test.
        """
        x_eval = np.array(test, dtype=np.float32)
        x_eval = torch.from_numpy(x_eval).to(device)
        with torch.no_grad():
            test = self.model.decode(x_eval)
        test = self.transform(test.cpu())
        test = test.numpy()
        return test

    def genotype2phenotype(
        self, genotype: typing.List[float]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the phenotype of the generator.

        Args:
            genotype (list): Genotype of the generator.

        Returns:
            tuple: Phenotype of the generator.
        """
        test = self.decode_test(genotype)
        result = self.orig_gen.genotype2phenotype(test)
        return result

    def set_genotype(self, phenotype):
        """Set the phenotype of the generator.

        Args:
            phenotype (list): Phenotype of the generator.
        """
        self.vector = phenotype

    def get_phenotype(self):
        """Get the phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.vector

    def generate_random_test(self) -> (typing.List[float], bool):
        """Generate a random test.

        Returns:
            tuple: Generated test and a boolean indicating success.
        """
        self.vector = np.random.normal(loc=self.mean, scale=self.std, size=self.size)
        return self.vector, True

    def visualize_test(
        self,
        test: typing.List[float],
        save_path: str = "test",
        num: int = 0,
        title: str = "",
    ):
        """Visualize a test.

        Args:
            test (np.array): Test to visualize.
            save_path (str, optional): Path to save the visualization. Defaults to "test".
            num (int, optional): Number of the visualization. Defaults to 0.
            title (str, optional): Title of the visualization. Defaults to "".
        """
        if self.orig_gen is not None:
            x_eval = np.array(test, dtype=np.float32)
            x_eval = torch.from_numpy(x_eval).to(device)
            with torch.no_grad():
                test = self.model.decode(x_eval)
            test = self.transform(test.cpu())
            test = test.numpy()
            phenotype = self.orig_gen.genotype2phenotype(test)
            self.orig_gen.visualize_test(phenotype, save_path, num, title)
