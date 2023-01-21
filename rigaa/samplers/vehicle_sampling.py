
import logging as log
import numpy as np
from pymoo.core.sampling import Sampling
import config as cf
import time
from rigaa.utils.car_road import Map
from rigaa.solutions import VehicleSolution
from rigaa.rl_agents.vehicle_agent2 import generate_rl_road
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.vehicle_evaluate import evaluate_scenario
from rigaa.utils.road_validity_check import is_valid_road


def generate_top_random_road(map_size):
    """
    It generates 20 random roads, and then returns the best one
    
    :param map_size: The size of the map
    :return: The best scenario and the best fitness
    """
    top = 20
    best_scenario = []
    best_fitness = 0
    for i in range(top):
        scenario, fitness = generate_random_road(map_size)
        if fitness > best_fitness:
            best_scenario = scenario
            best_fitness = fitness
    return best_scenario


def generate_random_road(map_size):
    """
    It generates a random road topology
    """
    actions = list(range(0, 3))
    lengths = list(range(cf.vehicle_env["min_len"], cf.vehicle_env["max_len"]))
    angles = list(range(cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"]))

    valid = False

    while valid == False:  # ensures that the generated road is valid
        done = False
        test_map = Map(map_size)
        while not done:
            action = np.random.choice(actions)
            if action == 0:
                length = np.random.choice(lengths)
                done = not (test_map.go_straight(length))
            elif action == 1:
                angle = np.random.choice(angles)
                done = not (test_map.turn_right(angle))
            elif action == 2:
                angle = np.random.choice(angles)
                done = not (test_map.turn_left(angle))
        scenario = test_map.scenario[:-1]

        road_points = test_map.get_points_from_states(scenario)
        intp_points = interpolate_road(road_points)
        valid = is_valid_road(intp_points)
        fitness, _ = evaluate_scenario(intp_points)
        fitness = abs(fitness)

    return scenario, fitness


class VehicleSampling(Sampling):

    """
    Module to generate the initial population

    returns: a tensor of candidate solutions
    """

    def __init__(self, init_pop_prob, map_size, executor):
        super().__init__()
        self.init_pop_prob = init_pop_prob
        self.map_size = map_size
        self.executor = executor
    def _do(self, problem, n_samples, **kwargs):

        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            r = np.random.random()
            if r < self.init_pop_prob:
                start = time.time()
                states, valid = generate_rl_road(self.map_size)
                if not(valid):
                    states = generate_top_random_road(self.map_size)
                log.debug("Individual produced by RL in %f sec", time.time() - start)
            else:
                start = time.time()
                states = generate_top_random_road(self.map_size)
                log.debug("Individual produced by randomly in %f sec", time.time() - start)
            s = VehicleSolution(self.map_size, self.executor)
            s.states = states
            X[i, 0] = s

        log.debug("Initial population of %d solutions generated", n_samples)
        return X
