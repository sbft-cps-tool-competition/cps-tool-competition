import numpy as np

def calculate_angle(p1, p2, p3):
    """
    The function calculates the angle between three points in a 2D space.
    
    :param p1: The parameter p1 represents the coordinates of the first point
    :param p2: The parameter p2 represents the vertex of the angle
    :param p3: The parameter p3 represents the coordinates of the third point in a 2D or 3D space. It
    can be a tuple or list containing the x, y, and z coordinates of the point
    :return: the angle between the vectors formed by points p1, p2, and p3 in degrees.
    """
    vector1 = np.array(p2) - np.array(p1)
    vector2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if magnitudes == 0:
        return 0  # Avoid division by zero
    angle = np.degrees(np.arccos(dot_product / magnitudes))
    return angle

def detect_turns(coordinates, threshold_angle):
    """
    The function "detect_turns" takes a list of coordinates and a threshold angle as input, and returns
    two lists - one containing the indices of the coordinates where turns occur, and another containing
    the indices of the coordinates where straight lines occur.
    
    :param coordinates: The coordinates parameter is a list of tuples representing the x and y
    coordinates of points in a path. Each tuple should have two elements, the x-coordinate and the
    y-coordinate, in that order. For example, [(0, 0), (1, 1), (2, 2)]
    :param threshold_angle: The threshold_angle parameter is the minimum angle (in degrees) that is
    considered a turn. Any angle greater than this threshold will be classified as a turn, while angles
    less than or equal to the threshold will be classified as straight segments
    :return: two lists: "turns" and "straights".
    """
    turns = []
    straights = []

    for i in range(1, len(coordinates) - 1):
        angle = calculate_angle(coordinates[i - 1], coordinates[i], coordinates[i + 1])
        if angle > threshold_angle:
            turns.append(i)
        else:
            straights.append(i)

    return turns, straights