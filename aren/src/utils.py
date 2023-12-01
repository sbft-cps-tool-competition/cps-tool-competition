
import math

from code_pipeline.validation import min_radius
from shapely.geometry import LineString

# Simplification: use a fixed d for all points
# TODO important: To increase granularity, reduce d_to_next_point, and adjust bounds for theta
D_TO_NEXT_POINT = 10


def getDirCov(x):    
    # theoretical maximum is 0.5, if we have a full circle.
    # This is because we use a method based on cos/dot-product
    angles_range = [0,0]
    cur_angle = 0
    num_p = x['num_points']
    for i in range(num_p):
        theta = x[f'p{i}_theta']
        cur_angle += theta
        angles_range[0] = min(angles_range[0], cur_angle)
        angles_range[1] = max(angles_range[1], cur_angle)

    dir_cov = (angles_range[1] - angles_range[0]) / 360
    return dir_cov


def getMaxCurv(x):

    max_curv = 0
    return max_curv


# ### VALIDITY HEURISTICS

TSHD_RADIUS=47
def heu_tooSharpTurns(test):
    min_r = min_radius(test.interpolated_points)
    if min_r > TSHD_RADIUS:
        return 0
    else:
        return TSHD_RADIUS - min_r


def heu_approxSelfIntersecting(points):
    # TODO improve granularity?
    line = LineString(points)
    return 0 if line.is_simple else 1

def heu_selfIntersecting(test):
    #  VERY time-consuming
    road_polygon = test.get_road_polygon()
    check = road_polygon.is_valid()
    return 0 if check else 1


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


INITIAL_POINT = (7, 7)
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
        # TODO better missing_dist?
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