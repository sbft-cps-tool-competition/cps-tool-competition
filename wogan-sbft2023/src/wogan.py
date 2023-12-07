from heapq import *
import logging as log

import numpy as np
import torch

from code_pipeline.tests_generation import RoadTestFactory

from src.sut import SUT
from src.models import WGAN


class WOGAN:

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

        # Use 20% of the execution budget for random generation. If the old time
        # budget is used, use 20% of it instead.
        if self.executor.time_budget.time_budget is None:
            self.init_budget = int(0.25 * self.executor.time_budget.execution_budget)
            self.execution_budget = self.executor.time_budget.execution_budget
        else:
            self.init_budget = int(0.25 * self.executor.time_budget.time_budget)
            self.execution_budget = self.executor.time_budget.time_budget

        # All code is from WOGAN commit d9caf426290df776e75a6f8ea8856552fddc12af

        # All of this is from config.py
        # -------------------------------------------------------------------------
        noise_dim = 10
        gan_neurons = 128
        gan_learning_rate = 0.0001
        analyzer_learning_rate = 0.001
        analyzer_neurons = 32
        gp_coefficient = 10
        batch_size = 32
        train_settings_init = {"epochs": 3,
                               "analyzer_epochs": 20,
                               "critic_epochs": 5,
                               "generator_epochs": 1}
        train_settings = {"epochs": 2,
                          "analyzer_epochs": 10,
                          "critic_epochs": 5,
                          "generator_epochs": 1}
        # -------------------------------------------------------------------------

        # All of this is from algorithms.py method main_wgan
        # -------------------------------------------------------------------------
        # How much to decrease the target fitness per each round when selecting a
        # new generated test.
        self.fitness_coef = 0.95
        # How often to train.
        self.train_delay = 3
        # How many bins are used.
        self.bins = 10

        self.get_bin = lambda x: int(x * self.bins) if x < 1.0 else self.bins - 1
        self.test_bins = {i: [] for i in range(self.bins)}  # a dictionary to tell which test is in which bin
        self.tests_generated = 0  # how many tests are generated so far
        # -------------------------------------------------------------------------

        # Initialize the model.
        # Almost all of this is from config.py
        # -------------------------------------------------------------------------
        sut = SUT()
        sut.ndimensions = 6  # Rotation added

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device '{}'.".format(device))

        self.model = WGAN(sut=sut, validator=None, device=device, logger=log)

        self.model.train_settings_init = train_settings_init
        self.model.train_settings = train_settings
        self.model.noise_dim = noise_dim
        self.model.gan_neurons = gan_neurons
        self.model.gan_learning_rate = gan_learning_rate
        self.model.analyzer_learning_rate = analyzer_learning_rate
        self.model.analyzer_neurons = analyzer_neurons
        self.model.gp_coefficient = gp_coefficient
        self.model.batch_size = batch_size

        self.model.initialize()
        # -------------------------------------------------------------------------

        # We guess the sizes of the numpy arrays and enlarge if needed. We assume an
        # average execution time of 10 s.
        guess = int(self.execution_budget // 10)
        self.test_inputs = np.zeros(shape=(guess, self.model.sut.ndimensions))
        self.test_outputs = np.zeros(shape=(guess, 1))

    def pad_with_zeros(self, X, Y):
        K = 10
        new_X = np.zeros(shape=(X.shape[0] + K, X.shape[1]))
        new_Y = np.zeros(shape=(Y.shape[0] + K, Y.shape[1]))
        new_X[:X.shape[0]] = X
        new_Y[:Y.shape[0]] = Y
        return new_X, new_Y

    def get_elapsed_execution_time(self):
        if self.executor.time_budget.time_budget is None:
            return self.execution_budget - self.executor.time_budget.get_remaining_time()["execution-budget"]
        else:
            return self.execution_budget - self.executor.time_budget.get_remaining_time()["time-budget"]

    def bin_sample(self, N, S, shift):
        """
    Samples N bin indices. The distribution on the indices is defined as
    follows. Suppose that S is a nonnegative function satisfying
    S(-x) = 1 - x for all x. Consider the middle points of the bins. We map
    the middle point of the middle bin to 0 and the remaining middle points
    symmetrically around 0 with first middle point corresponding to -1 and the
    final to 1. We then shift these mapped middle points to the right by the
    given amount. The weight of the bin will is S(x) where x is the mapped and
    shifted middle point.
    """

        # If the number of bins is odd, then the middle point of the middle bin
        # interval is mapped to 0 and otherwise the point common to the two middle
        # bin intervals is mapped to 0.
        if self.bins % 2 == 0:
            h = lambda x: x - (int(self.bins / 2) + 0.0) * (1 / self.bins)
        else:
            h = lambda x: x - (int(self.bins / 2) + 0.5) * (1 / self.bins)

        # We basically take the middle point of a bin interval, map it to [-1, 1]
        # and apply S on the resulting point to find the unnormalized bin weight.
        weights = np.zeros(shape=(self.bins))
        for n in range(self.bins):
            weights[n] = S(h((n + 0.5) * (1 / self.bins)) - shift)
        # Normalize weights.
        weights = (weights / np.sum(weights))

        # Binning fix
        reversed_bins = list(self.test_bins.values())[::-1]
        empty_bins = 0
        for bin_, value in enumerate(reversed_bins):
            if len(value) == 0:
                continue
            else:
                empty_bins = bin_
                break

        reversed_weights = weights[::-1]
        l = np.sum(reversed_weights[empty_bins:]) ** -1

        for bin_, weight in enumerate(reversed_weights):
            if bin_ < empty_bins:
                reversed_weights[bin_] = 0
            elif bin_ >= empty_bins:
                reversed_weights[bin_] = l * weight

        weights = reversed_weights[::-1]

        idx = np.random.choice(list(range(self.bins)), N, p=weights)
        return idx

    def training_sample(self, N, X, Y, B, S, shift):
        """
    Samples N elements from X and corresponding values of Y. The sampling is
    done by picking a bin and uniformly randomly selecting a test from the bin,
    but we do not select the same test twice. The probability of picking each
    bin is computed via the function bin_sample.
    """

        sample_X = np.zeros(shape=(N, X.shape[1]))
        sample_Y = np.zeros(shape=(N, Y.shape[1]))
        available = {n: v.copy() for n, v in B.items()}
        for n, bin_idx in enumerate(self.bin_sample(N, S, shift)):
            # If a bin is empty, try one lower bin.
            while len(available[bin_idx]) == 0:
                bin_idx -= 1
                bin_idx = bin_idx % self.bins
            idx = np.random.choice(available[bin_idx])
            available[bin_idx].remove(idx)
            sample_X[n] = X[idx]
            sample_Y[n] = Y[idx]

        return sample_X, sample_Y

    def start(self):
        # Generate initial tests randomly.
        # ---------------------------------------------------------------------------
        log.info("Generating and running initial random valid tests.")
        self.generate_initial_random_tests()
        log.info("Finished generating and running {} initial random valid tests.".format(self.random_init))
        # ---------------------------------------------------------------------------

        # The shift for sampling training data. We increase the number linearly
        # according to the function R.
        a0 = 0  # initial value
        a1 = 3.0  # final value
        alpha = (a1 - a0) / (self.execution_budget - self.init_budget)
        # alpha = (a1-a0)/(session.N_tests - session.random_init)
        beta = a1 - alpha * self.execution_budget
        # beta = a1 - alpha*session.N_tests
        self.R = lambda x: alpha * x + beta
        self.S = lambda x: 1 / (1 + np.exp(-1 * x))

        # Train the model with initial tests.
        # ---------------------------------------------------------------------------
        log.info("Training the model with initial tests.")
        # Train the analyzer.
        for epoch in range(self.model.train_settings_init["analyzer_epochs"]):
            self.model.train_analyzer_with_batch(self.test_inputs[:self.random_init, :],
                                                 self.test_outputs[:self.random_init, :],
                                                 train_settings=self.model.train_settings_init,
                                                 log=True)
        # Train the WGAN with different batches.
        for epoch in range(self.model.train_settings_init["epochs"]):
            train_X, train_Y = self.training_sample(min(self.model.batch_size, self.random_init),
                                                    self.test_inputs[:self.random_init, :],
                                                    self.test_outputs[:self.random_init, :],
                                                    self.test_bins,
                                                    self.S,
                                                    self.R(self.get_elapsed_execution_time()))
            self.model.train_with_batch(train_X,
                                        train_Y,
                                        train_settings=self.model.train_settings_init,
                                        log=True)
        log.info("Finished training the model with initial tests.")
        # ---------------------------------------------------------------------------

        # Start the main generation
        # ---------------------------------------------------------------------------
        while self.executor.time_budget.can_run_a_test():
            log.info("Budget remaining: {}".format(self.executor.get_remaining_time()["time-budget"]))
            # Generate a new valid test with high fitness and decrease target fitness
            # as per execution of the loop.
            # -------------------------------------------------------------------------
            target_fitness = 0.95
            heap = []
            entry_count = 0
            found = False
            while not found:
                # Generate a test.
                candidate_test = self.model.generate_test(1)
                # Predict the candidate test's fitness.
                predicted_fitness = self.model.predict_fitness(candidate_test)[0, 0]
                # Add to heap.
                heappush(heap, (1 - predicted_fitness, entry_count, candidate_test[:]))
                entry_count += 1

                # Check if the best predicted test is good enough.
                if 1 - min(heap)[0] >= target_fitness:
                    # Execute the best predicted test.
                    best_fitness, _, best_test = heappop(heap)
                    log.info("Executing test {} with predicted fitness {}.".format(best_test, best_fitness))
                    real_fitness = self.execute(best_test)
                    log.info("Test outcome {}.".format(real_fitness))
                    # If the test was invalid, we continue to generate a new test.
                    if real_fitness < 0: continue
                    # Otherwise we save the test and its fitness and exit for training.
                    if self.tests_generated >= self.test_inputs.shape[0]:
                        self.test_inputs, self.test_outputs = self.pad_with_zeros(self.test_inputs, self.test_outputs)
                    self.test_inputs[self.tests_generated, :] = best_test
                    self.test_outputs[self.tests_generated, :] = real_fitness
                    self.test_bins[self.get_bin(self.test_outputs[self.tests_generated, 0])].append(
                        self.tests_generated)
                    self.tests_generated += 1
                    found = True
                else:
                    target_fitness *= self.fitness_coef

            # Train the model.
            # -------------------------------------------------------------------------
            if (self.tests_generated - self.random_init) % self.train_delay == 0:
                # Train the analyzer.
                analyzer_batch_size = self.tests_generated
                for epoch in range(self.model.train_settings["analyzer_epochs"]):
                    self.model.train_analyzer_with_batch(self.test_inputs[:self.tests_generated],
                                                         self.test_outputs[:self.tests_generated],
                                                         train_settings=self.model.train_settings,
                                                         log=True)
                # Train the WGAN.
                for epoch in range(self.model.train_settings_init["epochs"]):
                    # We include the new tests to the batch with high probability if and
                    # only if they have high fitness.
                    BS = min(self.model.batch_size, self.random_init)
                    train_X = np.zeros(shape=(BS, self.test_inputs.shape[1]))
                    train_Y = np.zeros(shape=(BS, self.test_outputs.shape[1]))
                    c = 0
                    for i in range(self.train_delay):
                        if self.get_bin(self.test_outputs[self.tests_generated - i - 1]) >= \
                                self.bin_sample(1, self.S, self.R(self.get_elapsed_execution_time()))[0]:
                            train_X[c] = self.test_inputs[self.tests_generated - i - 1]
                            train_Y[c] = self.test_outputs[self.tests_generated - i - 1]
                            c += 1
                    train_X[c:], train_Y[c:] = self.training_sample(BS - c,
                                                                    self.test_inputs[:self.tests_generated, :],
                                                                    self.test_outputs[:self.tests_generated, :],
                                                                    self.test_bins,
                                                                    self.S,
                                                                    self.R(self.get_elapsed_execution_time()))
                    self.model.train_with_batch(train_X,
                                                train_Y,
                                                train_settings=self.model.train_settings,
                                                log=True)
        # ---------------------------------------------------------------------------
        log.info("Generated total {} valid tests.".format(self.tests_generated))

    def execute(self, test):
        # Execute a test and return its fitness. We return -1 for invalid tests and errors."""

        # Convert to plane point and pass to executor.
        road_points = self.test_to_road_points(test.reshape(-1))
        # Execute the test.
        the_test = RoadTestFactory.create_road_test(road_points)
        test_outcome, description, execution_data = self.executor.execute_test(the_test)

        if execution_data:
            # New combined distance function, f = 0.75 * BOLP + 0.25 * D
            D = max((np.clip((2 - state.oob_distance), 0, 4) / 4) for state in execution_data)
            BOLP = max(state.oob_percentage for state in execution_data)
            return 0.75 * BOLP + 0.25 * D
            # return ((max((np.clip((2 - state.oob_distance),0 , 4)/4) for state in execution_data))*0.25 + (max(state.oob_percentage for state in execution_data))*0.75)
        else:
            # The test was invalid or some other error occurred.
            return -1

    def test_to_road_points(self, test):
        """
    Converts a test instance to road points.
    Args:
      test (list): List of length self.ndimensions of floats in [-1, 1].
    Returns:
      output (list): List of length self.ndimensions of coordinate tuples.
    """

        # This is from sut/sut_sbst.py

        # This is the same code as in the Frenetic algorithm.
        # https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/base_frenet_generator.py
        # We integrate curvature (acceleratation) to get an angle (speed) and then
        # we move one step to this direction to get position. The integration is
        # done using the trapezoid rule with step given by the first component of
        # the test. Previously the first coordinate was normalized back to the
        # interval [25, 35], now we simply fix the step size.
        step = 15

        # We undo the normalization of the curvatures from [-1, 1] to [-0.07, 0.07]
        # as in the Frenetic algorithm.

        # Road rotation angle (limited between -45 and 45 degrees)
        rotation = np.math.pi * test[0]

        # Curvature values
        curvature = 0.07 * test[1:]

        # The initial point is at the center of the map [100, 100]
        points = [(self.map_size / 2, self.map_size / 2)]

        # Adding the rotations to the curvature values
        angles = [np.math.pi / 2 + rotation]

        # Adding the second point.
        points.append((points[-1][0] + step * np.cos(angles[-1]), points[-1][1] + step * np.sin(angles[-1])))
        # Finding and adding the remaining points.
        for i in range(curvature.shape[0] - 1):
            angles.append(angles[-1] + step * (curvature[i + 1] + curvature[i]) / 2)
            x = points[-1][0] + step * np.cos(angles[-1])
            y = points[-1][1] + step * np.sin(angles[-1])
            points.append((x, y))

        return points

    def sample_input_space(self, N, curvature_points):
        """
    Return N samples (tests) from the input space.
    Args:
      N (int):                The number of tests to be sampled.
      curvature_points (int): Number of points on a road (test).
    Returns:
      tests (np.ndarray): Array of shape (N, curvature_points).
    """

        # This is from sut/sut_sbst.py

        # The components of the actual test are curvature values in the range
        # [-0.07, 0.07], but the generator output is expected to be in the interval
        # [-1, 1].
        #
        # We do not choose the components of a test independently in [-1, 1] but
        # we do as in the case of the Frenetic algorithm where the next component
        # is in the range of the previous value +- 0.05.
        tests = np.zeros(shape=(N, curvature_points))
        for i in range(N):
            tests[i, 0] = np.random.uniform(-1, 1)
            tests[i, 1] = np.random.uniform(-1, 1)
            for j in range(2, curvature_points):
                tests[i, j] = tests[i, j - 1] + (1 / 0.07) * np.random.uniform(-0.05, 0.05)
        return tests

    def generate_initial_random_tests(self):
        """Generate initial random tests."""

        # We use the initial time budget for random generation.
        if self.executor.time_budget.time_budget is None:
            condition = lambda: self.executor.get_remaining_time()[
                                    "execution-budget"] > self.executor.time_budget.execution_budget - self.init_budget
        else:
            condition = lambda: self.executor.get_remaining_time()[
                                    "time-budget"] > self.executor.time_budget.time_budget - self.init_budget

        while condition():
            log.info("Budget remaining: {}".format(self.executor.get_remaining_time()["time-budget"]))
            # Sample a new test.
            test = self.sample_input_space(1, self.model.sut.ndimensions)
            log.info("Executing test {}.".format(test))
            # Execute the test.
            output = self.execute(test)
            log.info("Test outcome {}.".format(output))
            # Value -1 signifies an invalid test.
            if output < 0: continue
            # Save the executed valid test.
            if self.tests_generated >= self.test_inputs.shape[0]:
                self.test_inputs, self.test_outputs = self.pad_with_zeros(self.test_inputs, self.test_outputs)
            self.test_inputs[self.tests_generated, :] = test
            self.test_outputs[self.tests_generated, :] = output
            self.tests_generated += 1

        self.random_init = self.tests_generated

        # Assign the initial tests to bins.
        for n in range(self.tests_generated):
            self.test_bins[self.get_bin(self.test_outputs[n, 0])].append(n)
