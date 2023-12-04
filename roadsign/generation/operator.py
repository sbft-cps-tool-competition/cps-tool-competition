from copy import deepcopy

import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling

from ..road import Road, random_road, segments_distance

def generate_diverse_roads(rng: np.random.Generator, segments_range, length_range, total_length_range, radius_range, map_size, gen_count, gen_candidates):
    generated = [random_road(rng=rng, segments_range=segments_range, length_range=length_range, total_length_range=total_length_range, radius_range=radius_range, map_size=map_size)]
    for _ in range(gen_count-1):
        best_candidate = None
        best_diversity = 0.0
        for _ in range(gen_candidates):
            candidate = random_road(rng=rng, segments_range=segments_range, length_range=length_range, total_length_range=total_length_range, radius_range=radius_range, map_size=map_size)
            diversity = min(segments_distance(candidate.segments, road.segments, length_range=length_range, radius_range=radius_range) for road in generated)
            if diversity > best_diversity:
                best_diversity = diversity
                best_candidate = candidate
        generated.append(best_candidate)
    return generated

class TestGenerationCrossover(Crossover):
    RETRIES = 20
    def __init__(self, prob=1.0):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.prob_crossover = prob
    def do(self, problem, pop, parents=None, **kwargs):
        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]
        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)
        # perform crossover
        children = list(self._do(problem, X, **kwargs))
        return Population.new("X", children)
    def _do(self, problem, X, **kwargs):
        pa, pb = deepcopy(X)
        n_matings = len(pa)
        matings = 0
        retries = 0
        cross = np.random.random(n_matings) < self.prob_crossover
        while matings < n_matings:
            if (not cross[matings]) or retries > TestGenerationCrossover.RETRIES:
                yield pa[matings]
                yield pb[matings]
                matings += 1
                retries = 0
                continue
            p0a, p0b = pa[matings].points[0], pb[matings].points[0]
            a, b = pa[matings].segments, pb[matings].segments
            ap = int(problem.rng.uniform(low=1, high=len(a)+1))
            bp = int(problem.rng.uniform(low=1, high=len(b)+1))
            count = ap + (len(b) - bp), bp + (len(a) - ap)
            if min(count) < problem.segments_range[0] or max(count) > problem.segments_range[1]:
                retries += 1
                continue
            offsprings = np.array((
                Road.from_segments_and_starting_point(
                    segments=[*a[:ap], *b[bp:]],
                    start_point=p0a,
                    start_angle=Road.angle_towards_center(point=p0a, map_size=problem.map_size),
                    map_size=problem.map_size,
                ),
                Road.from_segments_and_starting_point(
                    segments=[*b[:bp], *a[ap:]],
                    start_point=p0b,
                    start_angle=Road.angle_towards_center(point=p0b, map_size=problem.map_size),
                    map_size=problem.map_size,
                ),
            ))
            if not all((
                road.validate(segments_range=problem.segments_range, total_length_range=problem.total_length_range)
                for road in offsprings
            )):
                retries += 1
                continue
            yield offsprings[0]
            yield offsprings[1]
            matings += 1
            retries = 0

class TestGenerationMutation(Mutation):
    def __init__(self, prob_start=.1, prob_segments=.1, prob_segments_flip=.1, prob_segments_radius=.1, prob_segments_length=.1):
        super().__init__()
        self.prob_start = prob_start
        self.prob_segments = prob_segments
        self.prob_segments_flip = prob_segments_flip
        self.prob_segments_radius = prob_segments_radius
        self.prob_segments_length = prob_segments_length
    def _do(self, problem, X, **kwargs):
        X = deepcopy(X)
        for i in range(len(X)):
            road = deepcopy(X[i])
            start_point=road.points[0]
            mutated = False
            if problem.rng.uniform() < self.prob_start:
                start_point = Road.random_start_point(rng=problem.rng, map_size=problem.map_size)
                mutated = True
            if problem.rng.uniform() < self.prob_segments:
                for segment in road.segments:
                    if problem.rng.uniform() < self.prob_segments_flip:
                        segment.radius = -segment.radius
                    if problem.rng.uniform() < self.prob_segments_radius:
                        segment.radius = problem.rng.uniform(low=problem.radius_range[0], high=problem.radius_range[1]) * np.sign(segment.radius)
                    if problem.rng.uniform() < self.prob_segments_length:
                        segment.length = problem.rng.uniform(low=problem.length_range[0], high=problem.length_range[1])
                mutated = True
            if mutated:
                new_road = Road.from_segments_and_starting_point(
                    segments=road.segments,
                    start_point=start_point,
                    start_angle=Road.angle_towards_center(point=start_point, map_size=problem.map_size),
                    map_size=problem.map_size
                )
                if new_road.validate(total_length_range=problem.total_length_range):
                    X[i] = road
        return X

class TestGenerationSampling(Sampling):
    def __init__(self, gen_count=10, gen_candidates=16):
        super().__init__()
        self.gen_count = gen_count
        self.gen_candidates = gen_candidates
        self.samples = []
    def _do(self, problem, n_samples, **kwargs):
        return self.get_or_generate_samples(problem=problem, n_samples=n_samples)
    def get_or_generate_samples(self, problem, n_samples):
        while len(self.samples) < n_samples:
            self.samples.extend(generate_diverse_roads(
                rng=problem.rng,
                segments_range=problem.segments_range,
                length_range=problem.length_range,
                total_length_range=problem.total_length_range,
                radius_range=problem.radius_range,
                map_size=problem.map_size,
                gen_count=min(self.gen_count, n_samples - len(self.samples)),
                gen_candidates=self.gen_candidates,
            ))
        return self.samples
