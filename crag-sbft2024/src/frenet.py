import numpy as np
import numpy.random as ra


def frenet_to_cartesian_road_points(x0, y0, theta0, ds, kappas):
    road_points = [(x0, y0)]
    x = x0
    y = y0
    theta = theta0
    n = len(kappas) + 1
    for i in range(n):
        x = x + (ds * np.cos(theta))
        y = y + (ds * np.sin(theta))
        road_points.append((x, y))
        if i < n - 1:
            theta = theta + (kappas[i] * ds)
    return road_points


def frenet_to_cartesian_road_points_with_reframability_check(x0, y0, theta0, ds, kappas, lane_width, map_size):
    road_points = [(x0, y0)]
    x = x0
    y = y0
    theta = theta0
    n = len(kappas) + 1
    min_x = x
    min_y = y
    max_x = x
    max_y = y
    for i in range(n):
        x = x + (ds * np.cos(theta))
        y = y + (ds * np.sin(theta))
        road_points.append((x, y))
        if i < n - 1:
            theta = theta + (kappas[i] * ds)

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    is_reframable = (max_x - min_x <= map_size - 2 * lane_width) and (max_y - min_y <= map_size - 2 * lane_width)
    is_in_map = (max_x < map_size - lane_width) and (min_x > lane_width) and (max_y < map_size - lane_width) and (min_y > lane_width)
    return road_points, is_in_map, is_reframable, min_x, min_y


def divide_and_sample(min_value, max_value, n, i):
    """Divide the interval [min_value, max_value) to n smaller intervals
       and sample a uniformly distributed random number from the ith
       subinterval
    """
    size = (max_value - min_value) / n
    return min_value + ra.uniform(i * size, (i + 1) * size)


def random_road_points(lengths_indices, kappas_indices, param_value_count, max_segment_count, global_curvature_bound, ds):
    """Generate a random road using indices identifying parameter values."""
    x0 = 0
    y0 = 0
    theta0 = ra.uniform(0, 2 * np.pi)
    segment_counts = [int(divide_and_sample(1, max_segment_count + 1, param_value_count, i)) for i in lengths_indices]
    kappas = []
    for i in range(len(segment_counts)):
        segment_count = segment_counts[i]
        kappa = divide_and_sample(-global_curvature_bound, global_curvature_bound, param_value_count, kappas_indices[i])
        kappas += [kappa for j in range(segment_count)]
    return frenet_to_cartesian_road_points(x0, y0, theta0, ds, kappas)

def random_road_points_with_reframability_check(lengths_indices, kappas_indices, param_value_count, min_segment_count, max_segment_count, global_curvature_bound, ds, lane_width, map_size):
    """Generate a random road using indices identifying parameter values."""
    x0 = 0
    y0 = 0
    theta0 = ra.uniform(0, 2 * np.pi)
    segment_counts = [int(divide_and_sample(min_segment_count, max_segment_count + 1, param_value_count, i)) for i in lengths_indices]
    kappas = []
    for i in range(len(segment_counts)):
        segment_count = segment_counts[i]
        kappa = divide_and_sample(-global_curvature_bound, global_curvature_bound, param_value_count, kappas_indices[i])
        kappas += [kappa for j in range(segment_count)]
    return frenet_to_cartesian_road_points_with_reframability_check(x0, y0, theta0, ds, kappas, lane_width, map_size)
