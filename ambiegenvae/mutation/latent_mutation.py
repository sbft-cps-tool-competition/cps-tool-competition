from ambiegenvae.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random


class LatentMutation(AbstractMutation):
    '''
    A class representing a latent mutation for a genetic algorithm.

    This class implements various mutation operations on a latent vector. Here kappas refer to the values in the latent vector.

    Mutation ideas borrowed from:
    https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/random_frenet_generator.py

    Parameters:
    - mut_rate (float): The mutation rate, indicating the probability of mutation for each element in the latent vector.

    Methods:
    - _do_mutation(x): Performs a mutation operation on the given latent vector.
    - _random_modification(kappas): Randomly modifies a subset of kappas in the given latent vector.
    - _reverse_kappas(kappas): Reverses the order of the kappas in the given latent vector.
    - _split_and_swap_kappas(kappas): Splits the kappas in the given latent vector into two halves and swaps their positions.

    '''

    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)

    def _do_mutation(self, x) -> np.ndarray:
        '''
        Performs a mutation operation on the given latent vector.

        Parameters:
        - x (np.ndarray): The latent vector to be mutated.

        Returns:
        - np.ndarray: The mutated latent vector.
        '''
        possible_mutations = [
            self._random_modification,
            self._reverse,
            self._split_and_swap
        ]

        mutator = np.random.choice(possible_mutations)

        return mutator(x)
    
    def _random_modification(self, kappas: np.ndarray) -> np.ndarray:
        '''
        Randomly modifies a subset of kappas in the given latent vector.

        Parameters:
        - kappas (np.ndarray): The latent vector containing the kappas.

        Returns:
        - np.ndarray: The modified latent vector.
        '''
        # number of kappas to be modified
        k = random.randint(5, (kappas.size-2) )
        # Randomly modified kappa
        indexes = random.sample(range((kappas.size) - 1), k)
        modified_kappas = kappas[:]
        for i in indexes:
            modified_kappas[i] += random.choice(np.linspace(-0.05, 0.05))
            modified_kappas[i] = max(0.0, modified_kappas[i])
            modified_kappas[i] = min(1.0, modified_kappas[i])
        return modified_kappas
    
    def _reverse(self, kappas: np.ndarray) -> np.ndarray:
        '''
        Reverses the order of the kappas in the given latent vector.

        Parameters:
        - kappas (np.ndarray): The latent vector containing the kappas.

        Returns:
        - np.ndarray: The reversed latent vector.
        '''
        return kappas[::-1]
    
    def _split_and_swap(self, kappas: np.ndarray) -> np.ndarray:
        '''
        Splits the kappas in the given latent vector into two halves and swaps their positions.

        Parameters:
        - kappas (np.ndarray): The latent vector containing the kappas.

        Returns:
        - np.ndarray: The modified latent vector.
        '''
        return np.concatenate([kappas[int((kappas.size) / 2):], kappas[:int((kappas.size) / 2)]])