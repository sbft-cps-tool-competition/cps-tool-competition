from ambiegenvae.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random


class KappaMutation(AbstractMutation):
    """
    A class representing a mutation strategy for modifying kappa values.

    This class implements various mutation methods for modifying kappa values in an array.

    Methods:
    - _do_mutation(x): Performs a mutation on the given array of kappa values.
    - _increase_kappas(kappas): Increases the values of kappa by a random factor.
    - _random_modification(kappas): Randomly modifies a subset of kappa values.
    - _reverse_kappas(kappas): Reverses the order of kappa values.
    - _split_and_swap_kappas(kappas): Splits the array of kappa values in half and swaps the halves.
    - _flip_sign_kappas(kappas): Flips the sign of each kappa value.

    Attributes:
    - mut_rate: The mutation rate, indicating the probability of performing a mutation.

    References:
    - Some mutation ideas borrowed from: https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/random_frenet_generator.py
    """

    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)

    def _do_mutation(self, x) -> np.ndarray:
        """
        Performs a mutation on the given array of kappa values.

        Parameters:
        - x: The array of kappa values.

        Returns:
        - The mutated array of kappa values.
        """
        possible_mutations = [
            self._increase_kappas,
            self._random_modification,
            self._reverse_kappas,
            self._split_and_swap_kappas,
            self._flip_sign_kappas,
        ]

        mutator = np.random.choice(possible_mutations)

        return mutator(x)

    def _increase_kappas(self, kappas: np.ndarray) -> np.ndarray:
        """
        Increases the values of kappa by a random factor.

        Parameters:
        - kappas: The array of kappa values.

        Returns:
        - The array of kappa values with increased values.
        """
        return np.array(list(map(lambda x: x * np.random.uniform(1.1, 1.2), kappas)))

    def _random_modification(self, kappas: np.ndarray) -> np.ndarray:
        """
        Randomly modifies a subset of kappa values.

        Parameters:
        - kappas: The array of kappa values.

        Returns:
        - The array of kappa values with randomly modified values.
        """
        k = random.randint(5, (kappas.size) - 5)
        indexes = random.sample(range((kappas.size) - 1), k)
        modified_kappas = kappas[:]
        for i in indexes:
            modified_kappas[i] += random.choice(np.linspace(-0.05, 0.05))
        return modified_kappas

    def _reverse_kappas(self, kappas: np.ndarray) -> np.ndarray:
        """
        Reverses the order of kappa values.

        Parameters:
        - kappas: The array of kappa values.

        Returns:
        - The array of kappa values with reversed order.
        """
        return kappas[::-1]

    def _split_and_swap_kappas(self, kappas: np.ndarray) -> np.ndarray:
        """
        Splits the array of kappa values in half and swaps the halves.

        Parameters:
        - kappas: The array of kappa values.

        Returns:
        - The array of kappa values with swapped halves.
        """
        return np.concatenate(
            [kappas[int((kappas.size) / 2) :], kappas[: int((kappas.size) / 2)]]
        )

    def _flip_sign_kappas(self, kappas: np.ndarray) -> np.ndarray:
        """
        Flips the sign of each kappa value.

        Parameters:
        - kappas: The array of kappa values.

        Returns:
        - The array of kappa values with flipped signs.
        """
        return np.array(list(map(lambda x: x * -1.0, kappas)))
