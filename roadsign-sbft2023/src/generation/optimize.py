from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.generation.operator import TestGenerationSampling, TestGenerationCrossover, TestGenerationMutation
from src.generation.problem import TestGenerationProblem

POP_SIZE = 100
DEFAULT_TERMINATION = get_termination('n_gen', 100)


class RoadDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def is_equal(self, a, b):
        return a == b


def get_sampling():
    return TestGenerationSampling(gen_count=10, gen_candidates=16)


def get_crossover():
    return TestGenerationCrossover(prob=1.0)


def get_mutation():
    return TestGenerationMutation(
        prob_start=.1,
        prob_segments=.1,
        prob_segments_flip=.1,
        prob_segments_radius=.1,
        prob_segments_length=.1,
    )


def get_algorithm(sampling=get_sampling(), crossover=get_crossover(), mutation=get_mutation()):
    return NSGA2(
        pop_size=POP_SIZE,
        n_offsprings=10,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=RoadDuplicateElimination(),
    )


def get_problem(rng, segments_range, length_range, total_length_range, radius_range, map_size):
    return TestGenerationProblem(
        rng=rng,
        segments_range=segments_range,
        length_range=length_range,
        total_length_range=total_length_range,
        radius_range=radius_range,
        map_size=map_size,
    )


def optimize(problem, termination=DEFAULT_TERMINATION, algorithm=get_algorithm()):
    return minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=problem.rng.integers(2 ** 32),
        save_history=False,
        verbose=True,
        copy_algorithm=True,
        copy_termination=True,
    )


class Optimizer:

    def __init__(self, rng, segments_range, length_range, total_length_range, radius_range, map_size,
                 termination=DEFAULT_TERMINATION, sampling=None):
        self.problem = get_problem(rng, segments_range, length_range, total_length_range, radius_range, map_size)
        self.termination = termination
        if sampling is None:
            sampling = get_sampling()
        self.sampling = sampling
        self.algorithm = get_algorithm(sampling=sampling)

    def run(self):
        return optimize(problem=self.problem, termination=self.termination, algorithm=self.algorithm)
