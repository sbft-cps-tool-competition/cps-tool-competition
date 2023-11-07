
from stable_baselines3 import PPO
import numpy as np

from rigaa.rl_envs.vehicle_env import CarEnv
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.vehicle_evaluate import evaluate_scenario
from rigaa.utils.car_road import Map
import config as cf
from rigaa.utils.road_validity_check import is_valid_road
def generate_rl_road(map_size):
    model_save_path = "models\\rl_model.zip" #curvy
    model = PPO.load(model_save_path)

    environ = CarEnv()

    scenario_found = False
    i = 0
    while scenario_found == False:
        obs = environ.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1
        map = Map(map_size)
        scenario = environ.all_states[-1]
        points = map.get_points_from_states(scenario)
        intp_points = interpolate_road(points)
        max_fitness, _ = (evaluate_scenario(intp_points))
        max_fitness = abs(max_fitness)

        if (max_fitness > 2.8) or i > 10:

            scenario_found = True
            valid = is_valid_road(intp_points)
            i  = 0
            
    return scenario, valid