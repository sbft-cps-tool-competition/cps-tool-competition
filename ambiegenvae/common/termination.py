from pymoo.core.termination import Termination

class BeamNGTermination(Termination):
    """
    Termination condition based on the evaluation budget.


    """

    def __init__(self) -> None:
        super().__init__()

    def _update(self, algorithm):
        """
        Updates the termination condition based on the algorithm's progress.
        """

        return 1 if algorithm.problem.executor.beamng_executor.time_budget.is_over() else 0
        
