import sys
import heapq
import numpy as np
import numpy.random as ra


from code_pipeline.tests_generation import RoadTestFactory
import logging as log
from .test_suite_generator import TestSuiteGenerator
from . import frenet
from . import frenetic

def update_average(average_dict, key_list, new_value):
    key_tuple = tuple(key_list)
    if key_tuple in average_dict:
        old_average, n = average_dict[key_tuple]
        new_average = (old_average * n + new_value) / (n + 1)
        average_dict[key_tuple] = (new_average, n + 1)
        return new_average
    else:
        average_dict[key_tuple] = (new_value, 1)
        return new_value


def take_best(lst, fun, best_size):
    size = len(lst)
    return [lst[j] for j in sorted(list(range(size)), key=fun)[:best_size]]


class CRAG:
    """
    Combinatorial RoAd Generator
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

        self.BEST_RATIO = 0.1 # What percentage of previous test suite we want to use as seed
        self.MAX_STRENGTH = 5 # We stop increasing n in n-wise combinations when we reach n=MAX_STRENGTH
        self.ROAD_PARAM_COUNT = 10 # Road Piece Length values + Kappa values
        self.ROAD_PARAM_VALUE_COUNT = 5 # Possible indices for the parameters are 0, 1, 2, 3

        self.MAX_ROAD_SCALAR = 1.5
        self.MIN_ROAD_SCALAR = 0.5

        self.MIN_OOB_SAMPLE_SIZE = 5 # We take average of MIN_OOB_SAMPLE_SIZE number of smallest oob distances

        self.INVALID_EVAL_RESULT = 100  # Result when simulation result is INVALID
        self.NOT_REFRAMABLE_RESULT = 1000  # Result when the road is not reframable
        self.IS_LIKELY_SELF_INTERSECTING_RESULT = 1000  # Result when the road is likely self-intersecting

        self.LANE_WIDTH = 10
        R = 15 # 14.326
        N = 70
        self.DS = 2 * R * np.cos((N-2) * np.pi / (2 * N))
        if self.DS > self.map_size: # This is unimaginable, but let's cover in case there is a unit change
            self.DS = self.map_size / 100

        self.MAX_ROAD_LENGTH = map_size * self.MAX_ROAD_SCALAR
        self.MAX_SEGMENT_COUNT = int((self.MAX_ROAD_LENGTH * 2 / self.ROAD_PARAM_COUNT) / self.DS)  # Maximum count of pieces in one section
        self.MIN_ROAD_LENGTH = map_size * self.MIN_ROAD_SCALAR
        self.MIN_SEGMENT_COUNT = int((self.MIN_ROAD_LENGTH * 2 / self.ROAD_PARAM_COUNT) / self.DS)  # Minimum count of pieces in one section
        self.GLOBAL_CURVATURE_BOUND = 2 * np.pi / (N * self.DS)  # Maximum of the absolute value of curvature

        self.test_suite_generator = TestSuiteGenerator(self.ROAD_PARAM_COUNT, self.ROAD_PARAM_VALUE_COUNT)

    def random_road_conf(self):
        road_conf = []
        for j in range(self.ROAD_PARAM_COUNT):
            road_conf.append(ra.randint(self.ROAD_PARAM_VALUE_COUNT))

        return road_conf

    def eval_road_conf(self, road_conf):
        # Generate road points using road configuration
        d = len(road_conf) // 2
        lengths_indices = [road_conf[2*i] for i in range(d)]
        kappas_indices = [road_conf[2*i + 1] for i in range(d)]
        # lengths_indices = road_conf[0:d]
        # kappas_indices = road_conf[d:]
        # road_points = frenet.random_road_points(lengths_indices, kappas_indices,
        #                                         self.ROAD_PARAM_VALUE_COUNT,
        #                                         self.MAX_SEGMENT_COUNT,
        #                                         self.GLOBAL_CURVATURE_BOUND,
        #                                         self.DS)
        #
        # Check self-intersections and reframe if possible
        # if frenetic.is_likely_self_intersecting(road_points, self.LANE_WIDTH):
        #     return self.IS_LIKELY_SELF_INTERSECTING_RESULT
        #
        # is_reframble, min_xs, min_ys = frenetic.is_reframable(self.map_size, road_points, self.LANE_WIDTH)
        # if not is_reframble:
        #     return self.NOT_REFRAMABLE_RESULT
        # #
        # road_points = frenetic.reframe_road(road_points, min_xs, min_ys, self.LANE_WIDTH)
        road_points, is_in_map, is_reframable, min_x, min_y = frenet.random_road_points_with_reframability_check(lengths_indices, kappas_indices,
                                                                                                                 self.ROAD_PARAM_VALUE_COUNT,
                                                                                                                 self.MIN_SEGMENT_COUNT,
                                                                                                                 self.MAX_SEGMENT_COUNT,
                                                                                                                 self.GLOBAL_CURVATURE_BOUND,
                                                                                                                 self.DS,
                                                                                                                 self.LANE_WIDTH,
                                                                                                                 self.map_size)

        if (not is_in_map) and (not is_reframable):
            return self.NOT_REFRAMABLE_RESULT

        # Check self-intersections and reframe if possible
        if frenetic.is_likely_self_intersecting(road_points, self.LANE_WIDTH):
            return self.IS_LIKELY_SELF_INTERSECTING_RESULT

        if not is_reframable:
            return self.NOT_REFRAMABLE_RESULT

        if not is_in_map:
            road_points = frenetic.reframe_road(road_points, min_x, min_y, self.LANE_WIDTH)


        log.info("CRAG: Evaluating road configuration %s.", str(road_conf))
        log.info("  Passing road points %s", str(road_points))

        # Execute test with the generated road
        the_test = RoadTestFactory.create_road_test(road_points)
        test_outcome, description, execution_data = self.executor.execute_test(the_test)
        if execution_data:
            min_oob_distances = heapq.nsmallest(self.MIN_OOB_SAMPLE_SIZE, [getattr(x, 'oob_distance') for x in execution_data])
            return sum(min_oob_distances) / len(min_oob_distances) # Return average of MIN_OOB_SAMPLE_SIZE number of smallest oob distances
        else:
            return self.INVALID_EVAL_RESULT

    def start(self):
        average_result_dictionary = {}
        # all_results = []
        best_of_last = None
        n = 2
        while True:
            test_suite = self.test_suite_generator.get_test_suite(n, best_of_last)
            log.info("CRAG: Created test suite of strength %d with %d road configurations.", n, len(test_suite))

            if test_suite is None or len(test_suite) == 0:
                n = 2
                continue

            results = []
            for road_conf in test_suite:
                if self.executor.is_over():
                    return
                result = self.eval_road_conf(road_conf)
                log.info("CRAG:  Got result: %s", str(result))
                # all_results.append(result)
                results.append(update_average(average_result_dictionary, road_conf, result))

            best_of_last = take_best(test_suite, lambda i: results[i], int(len(test_suite) * self.BEST_RATIO))
            if n < self.MAX_STRENGTH:
                n = n + 1
            else:
                n = 2
                best_of_last = None


