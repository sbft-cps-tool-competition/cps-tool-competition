import numpy as np

from pymoo.core.problem import ElementwiseProblem

'''
Multi-Objective Black-Box Test Case Selection for Cost-Effectively Testing Simulation Models
Arrieta et al. 2018

Instability + Discontinuity + Growth to Infinity, based on the turning angles of the road.

Fitness values are normalized over road length, preventing roads with not much going on in large sections.
'''

def segment_angles(segments, step=10):
    # Memoize traversed road distance for each segment
    l = np.empty(len(segments))
    l[0] = segments[0].length
    for i in range(1, len(segments)):
        l[i] = l[i - 1] + segments[i].length
    li = 0
    # Prepare step count and output array
    length = sum(s.length for s in segments)
    k = int(length / step)
    a = np.empty(k)
    # Compute values for each step
    a[0] = 0
    for i in range(1, k):
        while l[li] < i * step:
            li += 1
        segment = segments[li]
        a[i] = a[0] + (step / segment.radius)
    return a

def instability(signal):
    return sum((signal[i] - signal[i - 1] for i in range(1, len(signal))))

def discontinuity(signal):
    return (
        max((
            max((
                min(abs(signal[i] - signal[i - dt]), abs(signal[i + dt] - signal[i]))
                for i in range(dt, len(signal) - dt)
            ))
            for dt in [1, 2, 3]
        ))
    )

def growth(signal):
    return np.max(np.abs(signal))

def on_segment(p, q, r):
    return all((
        (q[0] <= max(p[0], r[0])),
        (q[0] >= min(p[0], r[0])),
        (q[1] <= max(p[1], r[1])),
        (q[1] >= min(p[1], r[1])),
    ))

def orientation(p, q, r):
    return np.sign(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))

def lines_intersect(a, b):
    o1 = orientation(a[0], a[1], b[0])
    o2 = orientation(a[0], a[1], b[1])
    o3 = orientation(b[0], b[1], a[0])
    o4 = orientation(b[0], b[1], a[1])
    return any((
        (o1 != o2) and (o3 != o4),
        (o1 == 0) and on_segment(a[0], b[0], a[1]),
        (o2 == 0) and on_segment(a[0], b[1], a[1]),
        (o3 == 0) and on_segment(b[0], a[0], b[1]),
        (o4 == 0) and on_segment(b[0], a[1], b[1]),
    ))

def path_self_intersects(path):
    path = list(((p[0], p[1]) for p in path))
    start_lines = {}
    end_lines = {}
    for p1, p2 in zip(path, path[1:]):
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        line = (p1, p2)
        start_lines[p1] = line
        end_lines[p2] = line
    bucket = set()
    x_sorted = sorted(path, key=lambda p: p[0])
    for point in x_sorted:
        e = end_lines.get(point)
        if e is not None:
            bucket.discard(e)
        s = start_lines.get(point)
        if s is not None:
            if any(map(lambda l: lines_intersect(l, s), bucket)):
                return True
            bucket.add(s)
    return False

def is_valid(road):
    return not path_self_intersects(road.points)

class TestGenerationProblem(ElementwiseProblem):
    
    def __init__(self, rng: np.random.Generator, segments_range, length_range, total_length_range, radius_range, map_size):
        super().__init__(n_var=1, n_obj=3)
        self.rng = rng
        self.segments_range = segments_range
        self.length_range = length_range
        self.total_length_range = total_length_range
        self.radius_range = radius_range
        self.map_size = map_size
        self.i = 0

    def _evaluate(self, x, out, *args, **kwargs):
        if is_valid(x):
            angles = segment_angles(x.segments)
            length = x.length()
            out["F"] = (
                -instability(angles),
                -discontinuity(angles),
                -growth(angles),
            )
        else:
            out["F"] = 0.0, 0.0, 0.0
