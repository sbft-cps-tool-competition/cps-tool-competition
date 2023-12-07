
from abc import ABC, abstractmethod

class AbstractValidator(ABC):
    def __init__(self, map_size: int):
        self.map_size = map_size

    @abstractmethod
    def is_valid(self, test) -> (bool, str):
        """
        Abstract method to check the validity of a test.

        Parameters:
        test (any): The test to be validated.

        Returns:
        bool: True if the test is valid, False otherwise.
        str: A message describing the validation result.
        """
        pass

        