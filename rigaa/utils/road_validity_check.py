
import numpy as np

from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
from numpy.ma import arange

#import config as cf
from shapely.geometry import LineString, Polygon
import config as cf

rounding_precision = 3
interpolation_distance = 1
smoothness = 0
min_num_nodes = 20

def interpolate_test(the_test):
    """
        Interpolate the road points using cubic splines and ensure we handle 4F tuples for compatibility
    """
    old_x_vals = [t[0] for t in the_test]
    old_y_vals = [t[1] for t in the_test]

    # This is an approximation based on whatever input is given
    test_road_lenght = LineString([(t[0], t[1]) for t in the_test]).length
    num_nodes = int(test_road_lenght / interpolation_distance)
    if num_nodes < min_num_nodes:
        num_nodes = min_num_nodes

    assert len(old_x_vals) >= 2, "You need at leas two road points to define a road"
    assert len(old_y_vals) >= 2, "You need at leas two road points to define a road"

    if len(old_x_vals) == 2:
        # With two points the only option is a straight segment
        k = 1
    elif len(old_x_vals) == 3:
        # With three points we use an arc, using linear interpolation will result in invalid road tests
        k = 2
    else:
        # Otheriwse, use cubic splines
        k = 3

    pos_tck, pos_u = splprep([old_x_vals, old_y_vals], s= smoothness, k=k)

    step_size = 1 / num_nodes
    unew = arange(0, 1 + step_size, step_size)

    new_x_vals, new_y_vals = splev(unew, pos_tck)

    # Return the 4-tuple with default z and defatul road width
    return list(zip([round(v, rounding_precision) for v in new_x_vals],
                    [round(v, rounding_precision) for v in new_y_vals],
                    [-28.0 for v in new_x_vals],
                    [8.0 for v in new_x_vals]))
def is_too_sharp(the_test, TSHD_RADIUS=47):
    """
    If the minimum radius of the test is greater than the TSHD_RADIUS, then the test is too sharp

    Args:
      the_test: the input road topology
      TSHD_RADIUS: The radius of the circle that is used to check if the test is too sharp. Defaults to
    47

    Returns:
      the boolean value of the check variable.
    """
    if TSHD_RADIUS > min_radius(the_test) > 0.0:
        check = True
        #print("Too sharp")
    else:
        check = False
    return check



def is_valid_road(points):
    """
    If the road is not simple, or if the road is too sharp, or if the road has less than 3 points, or if
    the last point is not in range, then the road is invalid

    Args:
      points: a list of points that make up the road

    Returns:
      A boolean value.
    """
    #nodes = [[p[0], p[1]] for p in points]
    # intp = self.interpolate_road(the_test.road_points)

    in_range = is_inside_map(points, cf.vehicle_env["map_size"])
    #the_test = MyRoadTestFactory.create_road_test(points)
    #test_validator = TestValidator(cf.vehicle_env["map_size"])
    the_test = interpolate_test(points)


    

    road = LineString([(t[0], t[1]) for t in points])
    invalid = (
        (road.is_simple is False)
       or (is_too_sharp(the_test) is True)
        #or (test_validator.is_too_sharp(the_test) is True)
        or (len(points) < 3)
        or (in_range is False)
    )
    return not(invalid)

# some of this code was taken from https://github.com/se2p/tool-competition-av
def find_circle(p1, p2, p3):
    """
    The function takes three points and returns the radius of the circle that passes through them

    Args:
      p1: the first point
      p2: the point that is the center of the circle
      p3: the point that is the furthest away from the line

    Returns:
      The radius of the circle.
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return np.inf

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    # print(radius)
    return radius


def min_radius(x, w=5):
    """
    It takes a list of points (x) and a window size (w) and returns the minimum radius of curvature of
    the line segment defined by the points in the window

    Args:
      x: the x,y coordinates of the points
      w: window size. Defaults to 5

    Returns:
      The minimum radius of curvature of the road.
    """
    mr = np.inf
    nodes = x
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]
        radius = find_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
    if mr == np.inf:
        mr = 0

    return mr * 3.280839895  # , mincurv


def is_inside_map(interpolated_points, map_size):
    """
        Take the extreme points and ensure that their distance is smaller than the map side
    """
    xs = [t[0] for t in interpolated_points]
    ys = [t[1] for t in interpolated_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return 0 < min_x or min_x > map_size and \
            0 < max_x or max_x > map_size and \
            0 < min_y or min_y > map_size and \
            0 < max_y or max_y > map_size