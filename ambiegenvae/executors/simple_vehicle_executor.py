
import logging  # as log
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.executors.abstract_executor import AbstractExecutor
from ambiegenvae.common.vehicle_evaluate import evaluate_scenario

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130


class SimpleVehicleExecutor(AbstractExecutor):
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
        self.name = "SimpleVehicleExecutor"

    def _execute(self, test) -> float:
        """
        Executes the given test scenario and returns the fitness value.

        Args:
            test: The test scenario to be executed.

        Returns:
            The fitness value of the executed test scenario.
        """

        fitness, _ = evaluate_scenario(test)

        log.info(f"Fitness: {fitness}")

        return fitness
