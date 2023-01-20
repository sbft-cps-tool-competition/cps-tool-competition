import logging as log
from copy import deepcopy
from time import time_ns

import numpy as np

from code_pipeline.tests_generation import RoadTestFactory

from .generation.optimize import Optimizer, get_termination, POP_SIZE
from .road import Road, segments_distance

def min_distance(roads, length_range, radius_range):
    return min(min(segments_distance(
        roads[i].segments, roads[j].segments,
        length_range=length_range, radius_range=radius_range
    ) for j in range(i + 1, len(roads))
    ) for i in range(len(roads) - 1))

def sample_roads(roads, samples, initial_sample, length_range, radius_range, threshold=0.0):
    if len(roads) <= samples:
        return roads
    sampled = {initial_sample}
    for _ in range(samples-1):
        best_candidate = None
        best_diversity = threshold
        for index in range(len(roads)):
            if index not in sampled:
                diversity = min(
                    segments_distance(
                        roads[index].segments, roads[s].segments,
                        length_range=length_range, radius_range=radius_range
                    ) for s in filter(lambda s: s != index, sampled)
                )
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = index
        if best_candidate is not None:
            sampled.add(best_candidate)
    return [roads[i] for i in sampled]

class RoadSignGenerator():
    """
    Generates roads in two steps:

    1. Random diverse roads are seeded as an initial population.
    2. A genetic algorithm evolves the population in order to maximize the following features:
        * Instability
        * Discontinuity
        * Growth

    For Step 2, a signal is generated for each road, based on periodic samples of its facing angle.
    """

    def __init__(self, executor, map_size=200, segments_range=(10, 30), length_range=None, total_length_range=None, radius_range=(30, 100), gen_max_time=60, seed_samples=10, gen_samples=10):
        self.executor = executor
        # Set default values based on map_size
        if length_range is None:
            length_range = (map_size/20, map_size/4)
        if total_length_range is None:
            total_length_range = (map_size, map_size*4)
        self.rng = np.random.Generator(np.random.MT19937(seed=time_ns()))
        self.segments_range = segments_range
        self.length_range = length_range
        self.total_length_range = total_length_range
        self.radius_range = radius_range
        self.map_size = map_size
        #self.termination = get_termination('n_gen', 300)
        self.termination = get_termination('time', gen_max_time)
        self.seed_samples = seed_samples
        self.gen_samples = gen_samples

    def start(self):
            log.info("Starting RoadSignGenerator")
            while not self.executor.is_over():
                try:
                    # Initialize optimizer
                    optimizer = Optimizer(
                        rng=self.rng,
                        segments_range=self.segments_range,
                        length_range=self.length_range,
                        total_length_range=self.total_length_range,
                        radius_range=self.radius_range,
                        map_size=self.map_size,
                        termination=self.termination,
                    )
                    log.info("[DIVERSE_ROADS] Start")
                    seeded_roads = optimizer.sampling.get_or_generate_samples(
                        problem=optimizer.problem, n_samples=POP_SIZE,
                    )
                    seeded_roads = deepcopy(sample_roads(
                        roads=seeded_roads,
                        samples=self.seed_samples,
                        initial_sample=self.rng.integers(low=0, high=len(seeded_roads)),
                        length_range=self.length_range,
                        radius_range=self.radius_range,
                        threshold=0.0,
                    ))
                    threshold = min_distance(
                        roads=seeded_roads,
                        length_range=self.length_range,
                        radius_range=self.radius_range,
                    ) / 2
                    self.execute_roads(seeded_roads, prefix='[DIVERSE_ROADS] ')
                    log.info("[DIVERSE_ROADS] Done")
                    if self.executor.is_over():
                        break
                    log.info("[OPTIMIZED_ROADS] Start")
                    results = optimizer.run()
                    optimized_roads = [
                        Road.from_segments(rng=self.rng, segments=r.segments, map_size=self.map_size) for r in results.X
                    ]
                    if len(optimized_roads) > self.gen_samples:
                        optimized_roads = sample_roads(
                            roads=optimized_roads,
                            samples=self.gen_samples,
                            initial_sample=self.rng.integers(low=0, high=len(optimized_roads)),
                            length_range=self.length_range,
                            radius_range=self.radius_range,
                            threshold=threshold,
                        )
                    log.info(f"[OPTIMIZED_ROADS] Generated tests: {len(optimized_roads)}")
                    self.execute_roads(optimized_roads, prefix='[OPTIMIZED_ROADS] ')
                except:
                    raise #log.exception('Exception in RoadSignGenerator')
            log.info("Finished RoadSignGenerator")

    def execute_roads(self, roads, prefix=''):
        log.info(f"{prefix}Starting test execution")
        for i, road in enumerate(roads, start=1):
            log.info(f"{prefix}Executing test {i}/{len(roads)}")
            self.execute_road(road)
        log.info(f"{prefix}Finished test execution")

    def execute_road(self, road):
        points = list((float(p[0]), float(p[1])) for p in road.points)
        log.info("Generated test from road points %s", points)
        the_test = RoadTestFactory.create_road_test(points)
        test_outcome, description, execution_data = self.executor.execute_test(the_test)
        log.info("Test outcome: {}. Description: {}".format(test_outcome, description))
        log.info("Remaining Time: %s", str(self.executor.get_remaining_time()["time-budget"]))
        return execution_data
