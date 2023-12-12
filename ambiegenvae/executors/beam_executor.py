
import logging #as log
from ambiegenvae.validators.abstract_validator import AbstractValidator
from ambiegenvae.executors.abstract_executor import AbstractExecutor
from code_pipeline.tests_generation import RoadTestFactory
log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130

class BeamExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, beamng_executor, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)
        self.beamng_executor = beamng_executor
        self._name = "BeamExecutor"

    def _execute(self, test) -> float:
        """
        Executes a test scenario in the BeamNG simulator and returns the fitness score.

        Args:
            test (list): The test scenario to be executed.

        Returns:
            float: The fitness score of the executed test scenario.
        """
        test_list = []
        for i in test:
            test_list.append(list(i))

        fitness = 0

        #if self.beamng_executor.get_remaining_time()["time-budget"] > 60:

        the_test = RoadTestFactory.create_road_test(test_list)

        test_outcome, description, execution_data = self.beamng_executor.execute_test(the_test)

        log.info(f"Test outcome: {test_outcome}")


        if test_outcome != "INVALID":
            fitness = -max([i.oob_percentage for i in execution_data])

        log.info(f"Fitness: {fitness}")
        #else:
        #    log.info("Low time budget, skipping test.")
        
        
        return fitness



