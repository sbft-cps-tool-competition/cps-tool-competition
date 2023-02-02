import numpy as np

# Margin to the map boundary
MARGIN = 8


def euclidean_distance(a, b):
    return sum(map(lambda v: (v[1] - v[0]) ** 2, zip(a, b)))


def rotate_point(point, angle):
    x, y = point
    sin, cos = np.sin(angle), np.cos(angle)
    return np.array((x * cos - y * sin, x * sin + y * cos))
    # return np.array((x*cos + y*sin, -x*sin + y*cos))


def longest_substring_within_bounds(points, map_size, distance_fn=euclidean_distance):
    length = map_size - 2 * MARGIN
    distance = 0.0
    max_distance = 0.0
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    start, stop = 0, 1
    begin, end = start, stop
    offset = np.array((0, 0))
    while stop < len(points):
        xmin, xmax, ymin, ymax = min(x[start:stop]), max(x[start:stop]), min(y[start:stop]), max(y[start:stop])
        if (xmax - xmin) > length or (ymax - ymin) > length:
            distance -= distance_fn(points[start], points[start + 1])
            start += 1
        else:
            if distance > max_distance:
                max_distance = distance
                begin, end = start, stop
                offset = np.array((-xmin + MARGIN, -ymin + MARGIN))
            stop += 1
            distance += distance_fn(points[stop - 2], points[stop - 1])
    return begin, end, offset


class RoadSegment:
    """
    Negative radius for left turn, positive for right turn
    RoadSegment.points generates the next 2 points from p0 = (0,0)
    """

    def __init__(self, length, radius):
        self.length = length
        self.radius = radius

    def __repr__(self) -> str:
        return f'({self.length},{self.radius})'

    def __eq__(self, other) -> bool:
        return self.length == other.length and self.radius == other.radius

    def __hash__(self):
        return hash((self.length, self.radius))

    def angle(self):
        return self.length / self.radius

    def points(self):
        r = self.radius

        def for_angle(a):
            dx = np.cos(a) * r - r
            dy = np.sin(a) * r
            return np.array((dx, dy))

        angle = self.angle()
        p1 = for_angle(angle / 2)
        p2 = for_angle(angle)
        return p1, p2, angle


class Road:
    def __init__(self, segments, points):
        self.segments = segments
        self.points = points

    def __repr__(self) -> str:
        return str(self.segments)

    def __eq__(self, other) -> bool:
        return self.points[0] == other.points[0], self.segments == other.segments

    def __hash__(self):
        return hash(((self.points[0][0], self.points[0][1]), self.segments))

    def length(self):
        return sum((s.length for s in self.segments))

    def validate(self, segments_range=None, total_length_range=None):
        if segments_range is not None:
            if len(self.segments) < segments_range[0] or len(self.segments) > segments_range[1]:
                return False
        if total_length_range is not None:
            length = self.length()
            if length < total_length_range[0] or length > total_length_range[1]:
                return False
        return True

    @staticmethod
    def generate_points(start_point, start_angle, segments):
        points = np.empty((len(segments) * 2 + 1, 2))
        points[0] = np.array(start_point)
        current_point, current_angle = start_point, start_angle
        for i, segment in enumerate(segments):
            p1, p2, angle = segment.points()
            p1 = rotate_point(p1, angle=current_angle) + current_point
            p2 = rotate_point(p2, angle=current_angle) + current_point
            current_angle += angle
            current_point = p2
            points[2 * i + 1] = p1
            points[2 * i + 2] = p2
        return points

    @staticmethod
    def angle_towards_center(point, map_size):
        center_point = (map_size / 2, map_size / 2)
        return np.arctan2(center_point[0] - point[0], center_point[1] - point[1])

    @staticmethod
    def random_start_point(rng: np.random.Generator, map_size):
        return rng.uniform(low=MARGIN, high=map_size - MARGIN + 1, size=2)

    @staticmethod
    def from_segments(rng: np.random.Generator, segments, map_size):
        # Generate random starting point and angle
        start_point = Road.random_start_point(rng=rng, map_size=map_size)  # Random point within the map
        start_angle = Road.angle_towards_center(point=start_point, map_size=map_size)  # Angle always towards the center
        # Generate road
        return Road.from_segments_and_starting_point(segments=segments, start_point=start_point,
                                                     start_angle=start_angle, map_size=map_size)

    @staticmethod
    def from_segments_and_starting_point(segments, start_point, start_angle, map_size):
        # Generate all points from segments
        points = Road.generate_points(start_point=start_point, start_angle=start_angle, segments=segments)
        # Truncate road if needed
        begin, end, offset = longest_substring_within_bounds(points=points, map_size=map_size)
        if begin != 0 or end != len(points):
            # Truncate complete segments
            begin_segment = (begin + 1) // 2
            end_segment = (end - 1) // 2
            segments = segments[begin_segment:end_segment]
            # Truncate points (aligned to segments) and offset
            begin_point = begin_segment * 2
            end_point = begin_point + 2 * len(segments) + 1
            points = points[begin_point:end_point] + offset
        # assert len(points) == 2 * len(segments) + 1
        return Road(segments=segments, points=list(points))


def random_segments(rng: np.random.Generator, length_range, radius_range):
    while True:
        length = rng.uniform(low=length_range[0], high=length_range[1])
        radius = rng.uniform(low=radius_range[0], high=radius_range[1])
        direction = int(rng.uniform() < .5) * 2 - 1  # -1 or 1
        yield RoadSegment(length=length, radius=radius * direction)


def random_road(rng: np.random.Generator, segments_range, length_range, total_length_range, radius_range, map_size):
    # Loop until valid road is generated
    while True:
        # Generate random road
        count = int(rng.uniform(low=segments_range[0], high=segments_range[1]))
        segments = []
        total_length = 0
        for i, segment in enumerate(random_segments(rng=rng, length_range=length_range, radius_range=radius_range)):
            length = segment.length
            if total_length + length > total_length_range[1]:
                break
            total_length += length
            segments.append(segment)
            if i >= count and total_length >= total_length_range[0]:
                break
        road = Road.from_segments(segments=segments, rng=rng, map_size=map_size)
        # Validate road
        if road.validate(segments_range=segments_range, total_length_range=total_length_range):
            return road


def segments_distance(a, b, length_range, radius_range, distance_fn=euclidean_distance):
    # Swap so that `a` is the longest sequence
    if len(a) < len(b):
        a, b = b, a
    # Check for special case: `b` is empty
    if len(b) == 0:
        return float(len(a) != 0)  # 0 if both are empty, 1 if `a` is not empty
    # Compute distance of most similar subsequence with length `len(b)`
    features = lambda s: (s.length, abs(s.radius), int(s.radius < 0))
    max_distance = distance_fn((length_range[0], radius_range[0], 0), (length_range[1], radius_range[1], 1)) * len(b)
    return min((  # Shortest distance between `a` and `b` for all possible offsets
        sum((  # Sum of the distances between segments
            distance_fn(features(a[i + o]), features(b[i])) for i in range(len(b))
        )) for o in range(len(a) - len(b) + 1)  # Offset of the comparison between `a` and `b`
    )) / max_distance  # Normalize


def segments_distance_levenshtein(a, b, length_range, radius_range, distance_fn=euclidean_distance):
    """
    Levenshtein distance
    """
    features = lambda s: (s.length, abs(s.radius), int(s.radius < 0))
    max_distance = distance_fn((length_range[0], radius_range[0], 0), (length_range[1], radius_range[1], 1))
    a_size, b_size = len(a) + 1, len(b) + 1
    d = np.zeros((a_size, b_size))
    d[:, 0] = np.arange(a_size)
    d[0, :] = np.arange(b_size)
    for i in range(1, a_size):
        for j in range(1, b_size):
            cost = distance_fn(features(a[i - 1]), features(b[j - 1])) / max_distance
            d[i, j] = min(
                d[i - 1, j] + 1,  # deletion
                d[i, j - 1] + 1,  # insertion
                d[i - 1, j - 1] + cost  # substitution
            )
    return d[len(a), len(b)]
