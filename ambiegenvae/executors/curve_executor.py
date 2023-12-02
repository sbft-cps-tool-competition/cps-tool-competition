
import logging  # as log
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.executors.abstract_executor import AbstractExecutor
from ambiegenvae.common.road_validity_check import min_radius

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47


class CurveExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """

    def __init__(
        self,
        generator,
        test_validator: AbstractValidator = None,
        results_path: str = None,
    ):
        super().__init__(generator, test_validator, results_path)
        self._name = "CurveExecutor"

    def _execute(self, test) -> float:
        """
        Executes the given test scenario and returns the fitness value.

        Args:
            test: The test scenario to be executed.

        Returns:
            The fitness value calculated based on the minimum curve radius.
        """

        min_curve = min_radius(test)

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
        else:
            fitness = -1 / min_curve

        log.info(f"Fitness: {fitness}")
        # fitness = 0

        return fitness
