#
# This file contains the code originally developed for DeepHyperion for computing exemplary test features.
# Please refer to the following paper and github repo:
# 	Tahereh Zohdinasab, Vincenzo Riccio, Alessio Gambi, Paolo Tonella:
#   DeepHyperion: exploring the feature space of deep learning-based systems through illumination search.
#   ISSTA 2021: 79-90
#
#   Repo URL: https://github.com/testingautomated-usi/DeepHyperion
#

import logging as logger
import math

import numpy as np

import matplotlib.pyplot as plt

from shapely.geometry import Point
from code_pipeline.utils import pairwise

THE_NORTH = [0, 1]


####################################################################################
# Private Utility Methods
####################################################################################


def _calc_angle_distance(v0, v1):
    at_0 = np.arctan2(v0[1], v0[0])
    at_1 = np.arctan2(v1[1], v1[0])
    return at_1 - at_0


def _calc_dist_angle(points):
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx):
        return np.subtract(points[idx + 1], points[idx])

    n = len(points) - 1
    result = [None] * (n)
    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = _calc_angle_distance(a, b)
        distance = np.linalg.norm(b)
        result[i] = (angle, distance, [points[i + 1], points[i]])
    return result


# TODO Possibly code duplicate
def _define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    center = Point(cx, cy)

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return center, radius


####################################################################################
# Structural Features
####################################################################################

# Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
# to approximate the road direction. By default we use 36 bins to have bins of 10 deg each
def get_thetas(xs, ys, skip=1):
    """Transform x,y coordinates of points and return each segment's offset from x-axis in the range [np.pi, np.pi]"""
    xdiffs = xs[1:] - xs[0:-1]
    ydiffs = ys[1:] - ys[0:-1]
    thetas = np.arctan2(ydiffs, xdiffs)
    return thetas


# Fixed direction coverage as suggested in Issue #114 by stklk
def direction_coverage_klk(the_test, n_bins=36):
    if not isinstance(the_test, list):
        the_test = the_test.interpolated_points
    np_arr = np.array(the_test)
    thetas = get_thetas(np_arr[:, 0], np_arr[:, 1])
    coverage_buckets = np.linspace(-np.pi, np.pi, num=n_bins)
    covered_elements = set(np.digitize(thetas, coverage_buckets))
    dir_coverage = len(covered_elements) / len(coverage_buckets)
    return "DIR_COV", dir_coverage

# Visualization of direction coverage
def visualize_dir_cov(coverage_buckets, covered_elements):
    plt.plot([-1.25,1.25], [0,0], color='black')
    plt.plot([0,0], [-1.25,1.25], color='black')
    plt.plot(np.cos(np.radians(coverage_buckets)), np.sin(np.radians(coverage_buckets)), color='black')
    plt.scatter(np.cos(np.radians(covered_elements)), np.sin(np.radians(covered_elements)), color="green")
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.gca().set_aspect('equal')
    plt.show()

# Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
# to approximate the road direction. By default we use 36 bins to have bins of 10 deg each
def direction_coverage(the_test, n_bins=36, debug=False):
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins + 1)
    direction_list = []
    if not isinstance(the_test, list):
        the_test = the_test.interpolated_points

    for a, b in pairwise(the_test):
        # Compute the direction of the segment defined by the two points
        road_direction = [b[0] - a[0], b[1] - a[1]]
        # Compute the angle between THE_NORTH and the road_direction.
        # E.g. see: https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        unit_vector_1 = road_direction / np.linalg.norm(road_direction)
        angle = math.degrees(math.atan2(unit_vector_1[1], unit_vector_1[0]) - math.atan2(THE_NORTH[1], THE_NORTH[0])) % 360
        direction_list.append(round(angle,3))

    # Place observations in bins and get the covered bins without repetition
    covered_elements = set(np.digitize(direction_list, coverage_buckets))
    dir_coverage = len(covered_elements) / len(coverage_buckets)
    if debug:
        visualize_dir_cov(coverage_buckets, [round(x*360/n_bins,3) for x in list(covered_elements)])
    return "DIR_COV", dir_coverage


def max_curvature(the_test, w=5):
    if not isinstance(the_test, list):
        nodes = the_test.interpolated_points
    else:
        nodes = the_test
    min_radius = np.inf
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]

        # We care only about the radius of the circle here
        _, radius = _define_circle(p1, p2, p3)
        if radius < min_radius:
            min_radius = radius
    # Max curvature is the inverse of Min radius
    curvature = 1.0 / min_radius

    return "MAX_CURV", curvature


####################################################################################
# Behavioural Features
####################################################################################


#  Standard Deviation of Steering Angle accounts for the variability of the steering
#   angle during the execution
def sd_steering(execution_data):
    steering = []
    for state in execution_data:
        steering.append(state.steering)
    sd_steering = np.std(steering)
    return "STD_SA", sd_steering


#  Mean of Lateral Position of the car accounts for the average behavior of the car, i.e.,
#   whether it spent most of the time traveling in the center or on the side of the lane
def mean_lateral_position(execution_data):
    lp = []
    for state in execution_data:
        lp.append(state.oob_distance)

    mean_lp = np.mean(lp)
    return "MEAN_LP", mean_lp


def max_lateral_position(execution_data):
    lp = []
    for state in execution_data:
        lp.append(state.oob_distance)

    max_lp = np.max(lp)
    return "MAX_LP", max_lp


def compute_all_features(the_test, execution_data):
    features = dict()
    # Structural Features
    structural_features = [max_curvature, direction_coverage]

    # Behavioural Features
    behavioural_features = [sd_steering, mean_lateral_position, max_lateral_position]

    logger.debug("Computing structural features")
    for h in structural_features:
        key, value = h(the_test)
        features[key] = value

    # TODO Add minimum value here
    if len(execution_data) > 2:
        logger.debug("Computing output features")
        for h in behavioural_features:
            key, value = h(execution_data)
            features[key] = value

    return features
