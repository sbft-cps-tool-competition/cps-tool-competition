"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for modeling the vehicle trajectory in a given scenario
"""

import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
from numpy.ma import arange
import math
import matplotlib.pyplot as plt
from descartes import PolygonPatch

from ambiegenvae.common.lane_controller import LaneController
from ambiegenvae.common.kinematic_model import KinematicModel


def interpolate_road(road, int_factor=7):
    """
    It takes a list of points (road) and returns a list of points (nodes) that are evenly spaced
    along the road

    Args:
      road: a list of tuples, each tuple is a point on the road

    Returns:
      A list of tuples.
    """

    test_road = LineString([(t[0], t[1]) for t in road])

    length = test_road.length

    num_nodes = int(length)
    if num_nodes < 20:
        num_nodes = 20

    old_x_vals = [t[0] for t in road]
    old_y_vals = [t[1] for t in road]

    if len(old_x_vals) == 2:
        k = 1
    elif len(old_x_vals) == 3:
        k = 2
    else:
        k = 3
    f2, u = splprep([old_x_vals, old_y_vals], s=0, k=k)

    step_size = 1 / num_nodes * int_factor

    xnew = arange(0, 1 + step_size, step_size)

    x2, y2 = splev(xnew, f2)

    nodes = list(zip(x2, y2))

    return nodes


def build_tc(road_points, car_path, fitness, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    road_x = []
    road_y = []

    for p in road_points:
        road_x.append(p[0])
        road_y.append(p[1])

    ax.plot(car_path[0], car_path[1], "bo", label="Car path")

    ax.plot(road_x, road_y, "yo--", label="Road")

    top = 200
    bottom = 0

    road_poly = LineString([(t[0], t[1]) for t in road_points]).buffer(
        8.0, cap_style=2, join_style=2
    )
    road_patch = PolygonPatch(
        (road_poly), fc="gray", ec="dimgray"
    )  # ec='#555555', alpha=0.5, zorder=4)
    ax.add_patch(road_patch)

    ax.set_title("Test case fitenss " + str(fitness), fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim(bottom, top)
    plt.ioff()
    ax.set_xlim(bottom, top)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def get_angle(node_a, node_b):
    """
    It takes two points, and returns the angle between them

    Args:
        node_a: The first node
        node_b: the node that is being rotated

    Returns:
        The angle between the two nodes.
    """
    vector = np.array(node_b) - np.array(node_a)
    cos = vector[0] / (np.linalg.norm(vector))

    angle = math.acos(cos)

    if node_a[1] > node_b[1]:
        return -angle
    else:
        return angle


def evaluate_scenario(points):
    """
    The function `evaluate_scenario` evaluates the fitness of a given scenario by simulating a vehicle's
    path and calculating the maximum distance traveled.
    
    Args:
      points: The "points" parameter is a list of coordinates that represent a road or path. Each
    coordinate is a tuple of (x, y) values. The points represent waypoints along the road that the
    vehicle will follow.
      rl_train: A boolean flag indicating whether the evaluation is being done during RL training or
    not. Defaults to False
    
    Returns:
      a tuple containing the fitness value and a list of x and y coordinates representing the path taken
    by the vehicle.
    """

    tot_x = []
    tot_y = []
    points = interpolate_road(points)

    #if is_valid_road(points):

    init_pos = points[0]
    x0 = init_pos[0]
    y0 = init_pos[1]
    yaw0 = 0  # get_angle(points[1], points[0]) #0
    speed0 = 15  # 12
    waypoints = points
    vehicle = KinematicModel(x0, y0, yaw0, speed0)
    controller = LaneController(waypoints, speed0)
    done = False
    distance_list = [0]
    steering = 0
    count = 0
    dt = 0.7
    while not (done):
        x, y, yaw, speed = vehicle.x, vehicle.y, vehicle.yaw, vehicle.speed
        steering, speed, distance, done = controller.control(x, y, yaw, speed)
        vehicle.update(steering, 0.1, dt, speed)  # accel = 0.05, v0 = 12
        tot_x.append(vehicle.x)
        tot_y.append(vehicle.y)
        count += 1
        if count > 4:
            distance_list.append(distance)

    car_path = LineString(zip(tot_x, tot_y))
    if car_path.is_simple is False:
        fitness = 3
    else:
        fitness = max(distance_list)

    return -fitness, [tot_x[:-1], tot_y[:-1]]



