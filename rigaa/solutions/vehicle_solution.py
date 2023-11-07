

import matplotlib.pyplot as plt
import time
import logging as log
import config as cf
from rigaa.utils.car_road import Map
from rigaa.utils.vehicle_evaluate import evaluate_scenario
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.road_validity_check import is_valid_road
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
from descartes import PolygonPatch

from code_pipeline.tests_generation import RoadTestFactory

import os
class VehicleSolution:

    """
    This is a class to represent one individual of the genetic algorithm
    It also contains the methods to evaluate the fitness of the solution, novelty and build the image
    """

    def __init__(self, map_size, executor):

        self.road_points = []
        self.states = []

        self.map_size = map_size
        self.fitness = 0
        self.car_path = []
        self.novelty = 0
        self.intp_points = []
        self.just_fitness = 0
        self.executor = executor

    def eval_fitness(self):
        """
        The function takes a list of states (self.states) and converts them to a list of points
        (self.road_points).
        The function then takes the list of points and interpolates them to create a list of interpolated
        points (self.intp_points).
        The function then takes the list of interpolated points and executes them with the simplified system model
        The function then calculates the fitness of the individual.
        Returns:
          The fitness of the individual.
        """
        test_map = Map(self.map_size)
        #car = Car(self.speed, self.steer_ang, self.map_size)
        road_points = test_map.get_points_from_states(self.states)
        self.states = self.states[:len(road_points)].copy()
        if len(road_points) <= 2:
            self.fitness = 0
        else:
            self.intp_points = interpolate_road(road_points)
            self.fitness, self.car_path = evaluate_scenario(
                self.intp_points
            )

        self.road_points = road_points

        return self.fitness

    def eval_fitness_full(self):

        map = Map(self.map_size)
        road_points = map.get_points_from_states(self.states)
        test = interpolate_road(road_points)
        is_valid = is_valid_road(test)
        #print(is_valid)

        if (is_valid):
            the_test = RoadTestFactory.create_road_test(test)
            time.sleep(1)
            oob_list = []
            if self.executor.get_remaining_time()["time-budget"] >= 40:
                test_outcome, description, execution_data = self.executor.execute_test(the_test)
                log.info("Test outcome: {}. Description: {}".format(test_outcome, description))
                log.info("Remaining Time: %s", str(self.executor.get_remaining_time()["time-budget"]))
            
                oob_list = [i.oob_percentage for i in execution_data]
            if oob_list:
                fitness = -max(oob_list)
                print("oob", fitness)
            else:
                fitness = 0
        else:
            fitness = 0

        self.fitness = fitness

        return fitness


    def intersect(self, tc1, tc2):
        """
        Compute the intersection of two sets (two test cases)

        Args:
          state1: the first element to compare
          state2: the second element to compare

        Returns:
          The list of similar elements in the two test cases 
        """
        intersection = []
        tc_size  = min(len(tc1), len(tc2))
        for i in range(tc_size):
            if tc1[i][0] == tc2[i][0]:
                if tc1[i][0] == 0:
                    if abs(tc1[i][1] - tc2[i][1]) <= 5:
                        intersection.append(tc1[i])
                else:
                    if abs(tc1[i][2] - tc2[i][2]) <= 5:
                        intersection.append(tc1[i])
        return intersection
                
    def calculate_novelty(self, tc1, tc2):
        """
        > The novelty of two test cases is the proportion of states that are unique to each test case
        We implement novelty calculation according to Jaccard distance definition:
        intersection/(set1 size + set2 size - intersection)
        
        :param tc1: The first test case
        :param tc2: The test case that is being compared to the test suite
        :return: The novelty of the two test cases.
        """
        intersection = self.intersect(tc1, tc2)
        total_states = len(tc1) + len(tc2) - len(intersection)

        novelty = 1 - len(intersection) / total_states
        return -novelty

    @staticmethod
    def build_image(states, save_path="test.png"):
        """
        It takes a list of states, and plots the road and the car path

        Args:
          states: a list of tuples, each tuple is a state of the car.
          save_path: The path to save the image to. Defaults to test.png
        """
        map_size = cf.vehicle_env["map_size"]
        test_map = Map(map_size)
        road_points = test_map.get_points_from_states(states)
        intp_points = interpolate_road(road_points)

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        fitness, car_path = evaluate_scenario(intp_points)

        if len(car_path):
            ax.plot(car_path[0], car_path[1], "bo", label="Car path")

        ax.plot(road_x, road_y, "yo--", label="Road")

        top = map_size
        bottom = 0

        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(8.0, cap_style=2, join_style=2)
        road_patch = PolygonPatch((road_poly), fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)
        ax.add_patch(road_patch)


        ax.set_title("Test case fitenss " + str(fitness), fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom, top)
        plt.ioff()
        ax.set_xlim(bottom, top)
        ax.legend()
        fig.savefig(save_path)
        plt.close(fig)