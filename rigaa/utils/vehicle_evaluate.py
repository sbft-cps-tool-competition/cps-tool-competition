
import json

import numpy as np

from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
from numpy.ma import arange

import matplotlib.pyplot as plt 
from shapely.geometry import LineString, Polygon
from descartes import PolygonPatch
from rigaa.utils.car_road import Map

from rigaa.utils.lane_controller import LaneController
from rigaa.utils.kinematic_model import KinematicModel
from rigaa.utils.road_validity_check import is_valid_road



def interpolate_road(road):
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

        step_size = 1 / num_nodes *5

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

    road_poly = LineString([(t[0], t[1]) for t in road_points]).buffer(8.0, cap_style=2, join_style=2)
    road_patch = PolygonPatch((road_poly), fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)
    ax.add_patch(road_patch)

    ax.set_title("Test case fitenss " + str(fitness), fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim(bottom, top)
    plt.ioff()
    ax.set_xlim(bottom, top)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def evaluate_scenario(points):

    tot_x = []
    tot_y = []
    

    #test_validator = TestValidator(200)

    #the_test = RoadTestFactory.create_road_test(points)
    
    #is_valid, validation_msg = test_validator.validate_test(the_test)

    #print(validation_msg)

    if is_valid_road(points):
    #if is_valid:

        init_pos = points[0]
        x0 = init_pos[0]
        y0 = init_pos[1]
        yaw0 = 0
        speed0 = 15  # 12
        waypoints = points
        vehicle = KinematicModel(x0, y0, yaw0, speed0)
        controller = LaneController(waypoints)
        done = False
        distance_list = [0]
        steering = 0
        count = 0
        dt = 0.7
        while not(done):
            x, y, yaw, speed = vehicle.x, vehicle.y, vehicle.yaw, vehicle.speed
            steering, speed_command, distance, done = controller.control(x, y, yaw, speed)
            vehicle.update(steering, 0.1, dt)  #accel = 0.05, v0 = 12
            tot_x.append(vehicle.x)
            tot_y.append(vehicle.y)
            count += 1
            if count > 6:
                #if distance < 7.5:
                distance_list.append(distance)

            #build_tc(points, [tot_x, tot_y], max(distance_list))

        car_path = LineString(zip(tot_x, tot_y))
        if car_path.is_simple is False:
            distance_list2 = [min(3, i) for i in distance_list]
        else:
            distance_list2 = distance_list


        if (distance_list[:-1]):
            fitness = max(distance_list2[:-1])
        else:
            fitness = max(distance_list2)

        #print(distance_list)
        # print("Fitness:", fitness)
    else: 
        fitness = 0




    return -fitness, [tot_x[:-1], tot_y[:-1]]
   


if __name__ == "__main__":
    #path  = "07-01-2023_tcs_1_rigaa_vehicle\\07-01-2023-tcs.json"
    #path = "06-01-2023-tcs_full_rigaa_vehicle\\22-12-2022-tcs.json"
    #path = "23-12-2022_tcs3_rigaa_vehicle\\23-12-2022-tcs.json"
    #path = "23-12-2022_tcs3_random_vehicle\\23-12-2022-tcs.json"
    #path = "22-12-2022_tcs3_nsga2_vehicle\\22-12-2022-tcs.json"
    path = "16-01-2023_tcs_rigaa_vehicle\\16-01-2023-tcs.json"

    with open(path, "r") as f:
        tcs = json.load(f)

    #states = tcs["run1"]["5"]
    

    for i in range(len(tcs["run4"])):
        #i = 11
        states = tcs["run4"][str(i)]
        test_map = Map(200)
        
        road = test_map.get_points_from_states(states)
        points = interpolate_road(road)
        fitness, car_path = evaluate_scenario(points)
        print(fitness)
        build_tc(points, car_path, fitness, "test\\" + str(i) + ".png")

    

    '''
    init_pos = points[0]
    x0 = init_pos[0]
    y0 = init_pos[1]
    yaw0 = 0
    speed0 = 12
    waypoints = points
    print(len(waypoints))
    vehicle = KinematicModel(x0, y0, yaw0, speed0)
    controller = LaneController(waypoints)
    done = False
    tot_x = []
    tot_y = []
    distance_list = []
    steering = 0
    while not(done):
        x, y, yaw, speed = vehicle.x, vehicle.y, vehicle.yaw, vehicle.speed
        steering, speed_command, distance, done = controller.control(x, y, yaw, speed, steering)
        vehicle.update(steering, 0.08, 0.5)
        #print(x, y)
        tot_x.append(vehicle.x)
        tot_y.append(vehicle.y)
        distance_list.append(distance)
        #build_tc(points, [tot_x, tot_y], max(distance_list))
    print(len(tot_x))
    fitness = max(distance_list)
    print(distance_list)
    print("Fitness:", fitness)
    build_tc(points, [tot_x, tot_y], fitness)
    '''


    
    