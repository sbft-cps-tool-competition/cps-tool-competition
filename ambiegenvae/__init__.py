from ambiegenvae.crossover.one_point_crossover import OnePointCrossover
from pymoo.operators.crossover.sbx import SBX

from ambiegenvae.mutation.kappa_mutations import KappaMutation
from ambiegenvae.mutation.latent_mutation import LatentMutation
from pymoo.operators.mutation.pm import PM


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES

from ambiegenvae.sampling.abstract_sampling import AbstractSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from ambiegenvae.sampling.greedy_sampling import GreedySampling

from ambiegenvae.problems.lkas_vae_problem import LKASVAEProblem
from ambiegenvae.problems.lkas_problem import LKASProblem

from ambiegenvae.executors.beam_executor import BeamExecutor
from ambiegenvae.executors.simple_vehicle_executor import SimpleVehicleExecutor
from ambiegenvae.executors.curve_executor import CurveExecutor

ALGORITHMS = {
    "ga": GA, # Genetic Algorithm,
    "de": DE, # Differential Evolution
    "es": ES # Evolution Strategy
}

SAMPLERS = {
    "random": FloatRandomSampling,
    "lhs": LHS,
    "abstract": AbstractSampling,
    "greedy": GreedySampling
}

CROSSOVERS = {
    "one_point": OnePointCrossover,
    "sbx": SBX
}

MUTATIONS = {
    "kappa": KappaMutation,
    "latent": LatentMutation,
    "pm": PM
}


PROBLEMS = {
    "lkas": LKASProblem,
    "lkasvae": LKASVAEProblem
}

EXECUTORS = {
    "beam": BeamExecutor,
    "simple_vehicle": SimpleVehicleExecutor,
    "curve": CurveExecutor
}







