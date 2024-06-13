from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer

growth = [
    (60, 25),
    (40, 40),
    (20, 95),
    (40, 150),
    (100, 175),
    (150, 160),
    (170, 125),
    (170, 75),
    (150, 50),
    (100, 50),
    (75, 75),
    (100, 120),
]
instability = [
    (13, 85),
    (38, 100),
    (63, 85),
    (88, 100),
    (113, 85),
    (138, 100),
    (163, 85),
    (188, 100),
]
discontinuity = [
    (10, 75),
    (30, 75),
    (50, 75),
    (70, 100),
    (100, 150),
    (130, 100),
    (150, 75),
    (170, 75),
    (190, 75),
]
points = list((float(p[0]), float(p[1])) for p in discontinuity)
the_test = RoadTestFactory.create_road_test(points)
RoadTestVisualizer(200).visualize_road_test(the_test)
