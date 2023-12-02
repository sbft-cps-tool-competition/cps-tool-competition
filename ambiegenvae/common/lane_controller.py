"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for controlling the vehicle to follow the waypoints
"""


import math
import numpy as np


class LaneController:
    """
    A class that represents a lane controller.

    Attributes:
        waypoints (list): The list of waypoints.
        current_waypoint (int): The index of the current waypoint.
        done (bool): Indicates if the lane controller has reached the end of the waypoints.
        max_steering (float): The maximum steering angle.
        previous_yaw (float): The previous yaw angle.
        window (int): The size of the window for searching waypoints.
        point_limit (int): The limit of waypoints to search within the window.
        speed_increment (float): The increment value for adjusting the speed.
        max_speed (float): The maximum speed.
        speed (float): The current speed.
        min_speed (float): The minimum speed.
    """

    def __init__(self, waypoints, speed):
        """
        Initializes a LaneController object.

        Args:
            waypoints (list): The list of waypoints.
            speed (float): The initial speed.
        """
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.done = False
        self.max_steering = math.pi
        self.previous_yaw = 0
        self.window = 10
        self.point_limit = 10
        self.speed_increment = 0.2
        self.max_speed = 30
        self.speed = speed
        self.min_speed = 8

    def control(self, x, y, yaw, speed):
        """
        Controls the lane based on the current position and speed.

        Args:
            x (float): The x-coordinate of the current position.
            y (float): The y-coordinate of the current position.
            yaw (float): The current yaw angle.
            speed (float): The current speed.

        Returns:
            tuple: A tuple containing the steering angle, adjusted speed, closest distance to the waypoint, and the status of completion.
        """
        # Find the next waypoint
        closest_distance = float("inf")
        closest_waypoint = self.current_waypoint

        for i, waypoint in enumerate(
            self.waypoints[self.current_waypoint : self.point_limit]
        ):
            self.point_limit = min(
                len(self.waypoints) - 1, self.current_waypoint + self.window
            )
            distance = math.sqrt((x - waypoint[0]) ** 2 + (y - waypoint[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_waypoint = i + self.current_waypoint
        self.current_waypoint = closest_waypoint

        # Calculate the target yaw based on the waypoint
        if self.current_waypoint >= len(self.waypoints) - 4:
            self.done = True
            steering = 0
        else:
            dx = self.waypoints[self.current_waypoint + 1][0] - x
            dy = self.waypoints[self.current_waypoint + 1][1] - y

            target_yaw = math.atan2(dy, dx)
            if dy > 0:  # and dx < 0:
                target_yaw -= 2 * math.pi

            self.previous_yaw = target_yaw

            # Calculate the steering angle
            steering = target_yaw - yaw

            self.speed = speed

            if abs(target_yaw - yaw) < 0.4:  # if the vehicle is going straight
                self.speed += self.speed_increment
                if self.speed > self.max_speed:
                    self.speed = self.max_speed
            elif abs(target_yaw - yaw) > 1.2:
                self.speed -= self.speed_increment / 2
                if self.speed < self.min_speed:
                    self.speed = self.min_speed

            # Limit the steering angle
            if steering > math.pi:
                steering = steering - 2 * math.pi
            elif steering < -math.pi:
                steering = steering + 2 * math.pi

            steering = min(steering, self.max_steering)
            steering = max(steering, -self.max_steering)
        return steering, self.speed, closest_distance, self.done

    def get_angle(self, node_a, node_b):
        """
        Calculates the angle between two nodes.

        Args:
            node_a (tuple): The coordinates of the first node.
            node_b (tuple): The coordinates of the second node.

        Returns:
            float: The angle between the two nodes.
        """
        vector = np.array(node_b) - np.array(node_a)
        cos = vector[0] / (np.linalg.norm(vector))

        angle = math.acos(cos)

        if node_a[1] > node_b[1]:
            return -angle
        else:
            return angle
