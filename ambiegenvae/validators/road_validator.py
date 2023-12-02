
from typing import List
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.common.road_validity_check import is_valid_road



class RoadValidator(AbstractValidator):
    def __init__(self, map_size: int):
        """
        Initializes a RoadValidator object.

        Args:
            map_size (int): The size of the map.

        """
        super().__init__(map_size=map_size)

    
    def is_valid(self, test: List[float]) -> (bool, str):
        """
        Checks if the given road is valid.

        Args:
            test (List[float]): The road to be validated.

        Returns:
            (bool, str): A tuple containing a boolean indicating if the road is valid and a string with information about any invalidity.

        """
        valid, invalid_info = is_valid_road(test, self.map_size)
        return valid, invalid_info
