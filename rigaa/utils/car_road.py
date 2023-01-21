#

import numpy as np
import math as m
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import logging as log

class Map:
    """Class that conducts transformations to vectors automatically,
    using the commads "go straight", "turn left", "turn right".
    As a result it produces a set of points corresponding to a road

    """

    def __init__(self, map_size):
        self.map_size = map_size
        self.width = 10
        self.max_x = map_size
        self.max_y = map_size
        self.min_x = 0
        self.min_y = 0
        self.radius = 15

        self.init_pos, self.init_end = self.init_position()

        self.road_point = []

        self.road_points_list = [[self.max_x / 2, self.max_y / 2]]

        self.scenario = []

        self.current_pos = [self.init_pos, self.init_end]
        self.all_position_list = [[self.init_pos, self.init_end]]

    def init_position(self):
        """
        It initializes the postion of the base vector to build the road

        Returns:
          The positions of start and end ponts of the road topology.
        """

        pos = np.array((self.max_y / 2 - self.width / 2, self.max_y / 2))
        end = np.array((self.max_y / 2 + self.width / 2, self.max_y / 2))

        return pos, end


    def position_to_center(self):
        """
        It takes the current position of the car, which is a list of two points, and returns the center of
        those two points

        Returns:
          The center of the road is being returned.
        """
        x = (self.current_pos[0][0] + self.current_pos[1][0]) / 2
        y = (self.current_pos[0][1] + self.current_pos[1][1]) / 2
        self.road_point = [x, y]
        return [x, y]

    def point_in_range(self, a):
        """check if point is in the acceptable range"""
        if ((4) < a[0] and a[0] < (self.max_x - 4)) and (
            (4) <= a[1] and a[1] < (self.max_y - 4)
        ):
            return 1
        else:
            return 0

    def go_straight(self, distance):
        """
        The function takes in the current position of the car, and the distance the car is supposed to
        move. It makes the parallel transition of the input vector at a given distance.

        Args:
          distance: the distance the car will travel

        Returns:
          a boolean value, True if the transformation was performed.
        """
        a = self.current_pos[0]
        b = self.current_pos[1]

        test_distance = 1

        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            return False

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        u_v = (p_a - p_b) / np.linalg.norm(p_b - p_a)
        sector = self.get_sector()

        if len(self.all_position_list) < 2:
            if sector == 0:
                R = np.array([[0, -1], [1, 0]])  # antilockwise
            elif sector == 1:
                R = np.array([[0, 1], [-1, 0]])  # clockwise
            elif sector == 2:
                R = np.array([[0, 1], [-1, 0]])
            elif sector == 3:
                R = np.array([[0, -1], [1, 0]])

            u_v_ = R.dot(u_v)

            p_a_ = p_a + u_v_ * distance
            p_b_ = p_b + u_v_ * distance

            self.current_pos = [p_a_, p_b_]
            self.all_position_list.append(self.current_pos)
            # return True
        else:
            R = np.array([[0, -1], [1, 0]])
            u_v_ = R.dot(u_v)
            p_a_ = p_a + u_v_ * test_distance  # make a small perturbation
            p_b_ = p_b + u_v_ * test_distance

            new_pos = [p_a_, p_b_]
            if self.in_polygon(new_pos) == True:  # check if it's in correct direction
                R = np.array([[0, 1], [-1, 0]])
                u_v = R.dot(u_v)
                p_a_ = p_a + u_v * distance
                p_b_ = p_b + u_v * distance
                self.current_pos = [p_a_, p_b_]
                self.all_position_list.append(self.current_pos)
                # return True
            else:
                p_a_ = p_a + u_v_ * distance
                p_b_ = p_b + u_v_ * distance
                self.current_pos = [p_a_, p_b_]
                self.all_position_list.append(self.current_pos)

        self.road_points_list.append(self.position_to_center())
        self.scenario.append([0, distance, 0])
        return True

    def get_sector(self):
        """returns the sector of initial position"""

        return 1

    def turn_right(self, angle):
        """
        The function turns the base vector to the right by the angle specified in the argument

        Args:
          angle: the angle of the turn

        Returns:
          the new position of the vector after it has turned right.
        """
        a = self.current_pos[0]
        b = self.current_pos[1]
        test_angle = 3
        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            return False

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        new_pos = self.clockwise_turn_top(test_angle, p_a, p_b)

        if self.in_polygon(new_pos) == True:
            self.current_pos = self.clockwise_turn_bot(angle, p_a, p_b)
        else:
            self.current_pos = self.clockwise_turn_top(angle, p_a, p_b)

        self.all_position_list.append(self.current_pos)

        self.road_points_list.append(self.position_to_center())
        self.scenario.append([1, 0, angle])
        return True

    def turn_left(self, angle):
        """
        The function takes in an angle and turns the vector to the left by that angle

        Args:
          angle: the angle of the turn

        Returns:
          the new position of the vector after it has turned left by the specified angle.
        """
        a = self.current_pos[0]
        b = self.current_pos[1]
        test_angle = 3
        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            return False

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        new_pos = self.anticlockwise_turn_top(test_angle, p_a, p_b)

        if self.in_polygon(new_pos) == True:
            self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
        else:
            self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
        self.all_position_list.append(self.current_pos)

        self.road_points_list.append(self.position_to_center())
        self.scenario.append([2, 0, angle])
        return True

    def clockwise_turn_top(self, angle, p_a, p_b):
        """
        Turns the input vector clockwise by the angle specified in the argument
        """
        angle += 180
        radius = self.radius

        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_a + u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)

        o_a_norm = np.linalg.norm(o_o - p_a)

        o_b = (o_o - p_b) / o_b_norm
        o_a = (o_o - p_a) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), np.sin(m.radians(angle))],
                [-np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def clockwise_turn_bot(self, angle, p_a, p_b):
        """
        Turns the input vector clockwise by the angle specified in the argument
        """
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_b - u_v * radius
        o_b_norm = np.linalg.norm(o_o - p_b)
        o_a_norm = np.linalg.norm(o_o - p_a)
        o_b = (p_b - o_o) / o_b_norm
        o_a = (p_a - o_o) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), np.sin(m.radians(angle))],
                [-np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )

        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm
        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def anticlockwise_turn_top(self, angle, p_a, p_b):
        """
        Turns the input vector anticlockwise by the angle specified in the argument"""
        angle += 180
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_a + u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)

        o_a_norm = np.linalg.norm(o_o - p_a)

        o_b = (o_o - p_b) / o_b_norm
        o_a = (o_o - p_a) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), -np.sin(m.radians(angle))],
                [np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def anticlockwise_turn_bot(self, angle, p_a, p_b):
        """
        Turns the input vector anticlockwise by the angle specified in the argument"""
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_b - u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)
        o_a_norm = np.linalg.norm(o_o - p_a)
        o_b = (p_b - o_o) / o_b_norm
        o_a = (p_a - o_o) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), -np.sin(m.radians(angle))],
                [np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def in_polygon(self, new_position):
        """checks whether a point lies within a polygon
        between current and previous vector"""
        if len(self.all_position_list) <= 1:
            return True
        current = self.all_position_list[-1]
        prev = self.all_position_list[-2]
        new = new_position
        new_mid = (new[0] + new[1]) / 2

        point = Point(new_mid[0], new_mid[1])
        polygon = Polygon(
            [tuple(current[0]), tuple(current[1]), tuple(prev[0]), tuple(prev[1])]
        )
        return polygon.contains(point)

    def get_points_from_states(self, states):
        """
        It takes a list of states, and for each state, it performs the action specified by the state, and
        then appends the resulting road points to a list

        Args:
          states: a list of tuples, each tuple is (action, angle, distance)

        Returns:
          The points of the road.
        """

        tc = states
        for state in tc:
            action = state[0]
            if action == 0:
                done = self.go_straight(state[1])
                if not (done):
                    break
            elif action == 2:
                done = self.turn_left(state[2])
                if not (done):
                    break
            elif action == 1:
                done = self.turn_right(state[2])
                if not (done):
                    break
            else:
                log.error("ERROR, invalid action")

        points = self.road_points_list[:-1]
        return points
