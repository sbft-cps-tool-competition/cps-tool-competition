
import math

from code_pipeline.validation import min_radius
from shapely.geometry import LineString

# NOTE, as simplification, we use a fixed d for all points

# TODO future work:
# fine-tune these values. For now, (20, 65, (3, 20)) is best, then (10, 35, (5, 30)), then (15, 50, (5, 30))
D_TO_NEXT_POINT = 20
THETA_MAX = 65
POINTS_RANGE = (3, 20)

TSHD_RADIUS=47
INITIAL_POINT = (7, 7)

# ### VALIDITY HEURISTICS

def heu_missing_distance(missing_dist, map_size):
    base = (map_size-2*INITIAL_POINT[0]) + (map_size-2*INITIAL_POINT[1])
    return missing_dist / base


def heu_approxSelfIntersecting(points):
    line = LineString(points)
    return 0 if line.is_simple else 1


# def heu_selfIntersecting(test):
#     # NOTE this is VERY time-consuming
#     road_polygon = test.get_road_polygon()
#     check = road_polygon.is_valid()
#     return 0 if check else 1


def heu_tooSharpTurns(test):
    min_r = min_radius(test.interpolated_points)
    if min_r > TSHD_RADIUS:
        return 0
    else:
        return (TSHD_RADIUS - min_r) / TSHD_RADIUS


# ### DIVERSITY HEURISTICS

class FeatureDistribution:

    def __init__(self, min_val, max_val):
        # TODO future work:
        # make this an ordered list to improve performance
        self.seen = []
        self.min = min_val
        self.max = max_val
        self.biggest_gap = max_val - min_val


def heu_diversity(feature_name, features, all_distributions):

    distribution = all_distributions[feature_name]
    feature_val = features[feature_name]

    # et closest point in both directions
    d_smallest, new_big_gap = diversity_metrics(distribution, feature_val)
    score = diversity_score(d_smallest, distribution.biggest_gap)

    # update the biggest gap
    if new_big_gap + d_smallest >= distribution.biggest_gap-0.00001:
        distribution.biggest_gap = new_big_gap

    # get heuristic from score

    # TODO future work:
    # this can beimproved, specially considering the aggregation applied to it
    # if score == 0:
    #     return float('inf'), score
    # else:
    #     return 1/(score*(1+len(distribution.seen))), score
    return (1-score), score


def diversity_metrics(distribution, new_value):

    # handle the (very unlikely) case where new_value is not in the range
    if new_value < distribution.min:
        distribution.min = new_value
    elif new_value > distribution.max:
        distribution.max = new_value

    # TODO future work:
    # do better handling of out-of-range cases
    
    upper_bound = distribution.max
    lower_bound = distribution.min
    
    smallest_dist_below = upper_bound-lower_bound
    smallest_dist_above = upper_bound-lower_bound
    for v in distribution.seen:
        # below
        d_below = new_value-v
        d_below = d_below if d_below >= 0 else upper_bound-lower_bound+(d_below) # circular
        smallest_dist_below = min(smallest_dist_below, d_below)

        # above
        d_above = d_below if d_below < 0.00001 else upper_bound-lower_bound-(d_below)
        smallest_dist_above = min(smallest_dist_above, d_above)

    if smallest_dist_below < smallest_dist_above:
        smallest_dist = smallest_dist_below
        new_biggest_poential_gap = smallest_dist_above
    else:
        smallest_dist = smallest_dist_above
        new_biggest_poential_gap = smallest_dist_below

    return smallest_dist, new_biggest_poential_gap


def diversity_score(small_dist, biggest_gap):
    # [(0)..(1)]
    # [(exactly on top of another point)..(exactly in the middle between the 2 furthest points)]
    return small_dist / (biggest_gap/2)


def heu_and_add_diversity(feature_name, features, all_distributions):
    
    heuristic, score = heu_diversity(feature_name, features, all_distributions)
    all_distributions[feature_name].seen.append(score)
    return heuristic, score


# PATH GENERATION + checking if inside region


def getRoadPointsFromAngles(x, map_size):
    p_pre, p_cur = getFirst2Points(x)

    road_points = [p_pre, p_cur]
    num_p = x['num_points']

    for i in range(num_p):
        theta = x[f'p{i}_theta']

        p_temp = p_cur
        p_cur = get_next_point(p_cur, p_pre, theta, D_TO_NEXT_POINT)
        p_pre = p_temp
        road_points.append(p_cur)

    # reframe the points into the map, if necessary
    road_points, missing_dist = reframe(road_points, map_size)
    return road_points, missing_dist


# NOTE: road width is 8 (lane width is 4)
def getFirst2Points(x):
    # Start at bottom-left corner, going diagonally to the top-right
    starting_angle = 45
    p0 = INITIAL_POINT
    
    p1x = p0[0] + D_TO_NEXT_POINT * math.cos(math.radians(starting_angle))
    p1y = p0[1] + D_TO_NEXT_POINT * math.sin(math.radians(starting_angle))
    p1 = (p1x, p1y)

    return p0, p1


def reframe(road_points, map_size):
    # find the span
    min_x, max_x = road_points[0][0], road_points[0][0]
    min_y, max_y = road_points[0][1], road_points[0][1]
    
    for point in road_points[1:]:
        x, y = point
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    x_transform = INITIAL_POINT[0]-min_x
    y_transform = INITIAL_POINT[1]-min_y

    if x_transform == 0 and y_transform == 0:
        # no need to reframe
        return road_points, 0
    
    # is it possible to refram the span?
    missing_x = max_x+x_transform - (map_size-INITIAL_POINT[0])
    missing_y = max_y+y_transform - (map_size-INITIAL_POINT[1])

    if missing_x < 0 and missing_y < 0:
        # size is ok, reframe
        reframed_points = []
        for (x, y) in road_points:
            reframed_points.append((x+x_transform, y+y_transform))

        return reframed_points, 0
    else:
        # size is too big, cannot reframe
        # TODO future work:
        # improveissing distance measurment?
        missing_dist = 0
        missing_dist += max(0, missing_x)
        missing_dist += max(0, missing_y)
        return None, missing_dist


def get_next_point(p_cur, p_prev, angle, dist):
    # Calculate the angle between p_prev and p_cur
    angle_rad = math.atan2(p_cur[1] - p_prev[1], p_cur[0] - p_prev[0])
    
    # Calculate the next point coordinates
    next_x = p_cur[0] + dist * math.cos(angle_rad + math.radians(angle))
    next_y = p_cur[1] + dist * math.sin(angle_rad + math.radians(angle))

    return (next_x, next_y)