
class SUT:
    """
    Base class implementing a system under test.
    """

    def __init__(self):
        self.ndimensions = None
        self.dataX = None
        self.dataY = None
        self.target = 1.0

    def execute_test(self, tests):
        raise NotImplementedError()

    def execute_random_test(self, N=1):
        raise NotImplementedError()

    def sample_input_space(self, N=1):
        raise NotImplementedError()
