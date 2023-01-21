

from gym import Env
from gym.spaces import Box, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config as cf
from rigaa.utils.car_road import Map
import os


class CarEnv(Env):
    def __init__(self):
        self.max_number_of_points = 30
        self.action_space = MultiDiscrete([3, cf.vehicle_env['max_len'] - cf.vehicle_env['min_len'], cf.vehicle_env['max_angle'] - cf.vehicle_env['min_angle']])  # 0 - increase temperature, 1 - decrease temperature
        
        self.max_steps = 29
        self.steps = 0

        self.max_fitness = 5

        self.fitness = 0

        self.all_results = {}       
        self.observation_space = Box(low=0, high=100, shape = (self.max_number_of_points*3, ), dtype=np.int8)

    def generate_init_state(self):

        road_type = np.random.randint(0, 2)
        value = np.random.randint(cf.vehicle_env["min_len"], cf.vehicle_env["max_len"] + 1)
        position = np.random.randint(cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"] + 1)
        state  = [road_type, value, position]
        road_type2 = np.random.randint(0, 2)
        value2 = np.random.randint(cf.vehicle_env["min_len"], cf.vehicle_env["max_len"] + 1)
        position2 = np.random.randint(cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"] + 1)
        state2  = [road_type2, value2, position2]
        return [state, state2]

    def set_state(self, action):
        map = Map(cf.vehicle_env['map_size'])
        if action[0] == 0:
            distance  = action[1] + cf.vehicle_env['min_len']
            self.done = not(map.go_straight(distance))
            angle = 0
        elif action[0] == 1:
            angle = action[2] + cf.vehicle_env['min_angle']
            self.done = not(map.turn_right(angle))
            distance = 0
        elif action[0] == 2:
            angle = action[2] + cf.vehicle_env['min_angle']
            self.done = not(map.turn_left(angle))
            distance = 0

        return [action[0], distance, angle]

    def step(self, action):

        assert self.action_space.contains(action)
        self.done = False
        
        self.state[self.steps] = self.set_state(action)

        reward = 0

        if self.done is True:
            reward = 5
        else:

            current_state = self.state[:self.steps].copy()
            #self.all_fitness.append(self.fitness)
            self.all_states.append(current_state)

        self.steps += 1

        if self.steps >= self.max_steps:

            self.done = True

        info = {}
        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8), reward, self.done, info


    def reset(self):
        #print("Reset")

        self.steps = 2
        #print(self.fitness)
        self.state = self.generate_init_state()#generate_random_state()#road()
        while len(self.state) < self.max_number_of_points:
            self.state.append([0, 0, 0])
        
        self.road = []
        self.car_path = []

        self.all_states = []
        self.all_fitness = []

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8)

    def get_size(self, states):
        size = 0
        for state in states:
            if state != [0, 0, 0]:
                size += 1
        return size

    def render(self, scenario, mode='human'):
        car = Car(cf.vehicle_env["speed"], cf.vehicle_env["steer_ang"], cf.vehicle_env["map_size"])
        size = self.get_size(scenario)
        map = Map(cf.vehicle_env['map_size'])
        points = map.get_points_from_states(scenario[:])
        intp_points = car.interpolate_road(points)

        reward, car_path = (car.execute_road(intp_points))
        reward = abs(reward)
        fitness = reward

        #if self.done:
        fig, ax = plt.subplots(figsize=(12, 12))
        road_x = []
        road_y = []
        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        ax.plot(road_x, road_y, "yo--", label="Road")


        if len(car_path):
            ax.plot(car_path[0], car_path[1], "bo", label="Car path")

        
        top = cf.vehicle_env["map_size"]
        bottom = 0

        ax.set_title(
            "Test case fitenss " + str(fitness) , fontsize=17
        )

        ax.set_ylim(bottom, top)

        ax.set_xlim(bottom, top)
        ax.legend()

        fig.savefig("test.png")

        plt.close(fig)