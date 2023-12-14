import numpy as np
import torch
from pymoo.optimize import minimize
import logging
from pymoo.termination import get_termination
from ambiegenvae.problems.lkas_vae_problem import LKASVAEProblem
from ambiegenvae.common.random_seed import get_random_seed
from ambiegenvae.executors.beam_executor import BeamExecutor
from ambiegenvae.validators.road_validator import RoadValidator
from ambiegenvae.generators.kappa_generator import KappaRoadGenerator
from ambiegenvae.common.duplicate_removal import AbstractDuplicateElimination

from ambiegenvae.common.get_convergence import get_convergence
from ambiegenvae.common.get_stats import get_stats
from ambiegenvae.common.get_test_suite import get_test_suite
from ambiegenvae.common.save_tc_results import save_tc_results
from ambiegenvae.generators.latent_generator import LatentGenerator
from ambiegenvae.common.train_vae import Denormalize1D, VecVAE
from ambiegenvae.common.test_vae import load_model
from ambiegenvae.common.termination import BeamNGTermination

from ambiegenvae import ALGORITHMS, SAMPLERS, CROSSOVERS, MUTATIONS

log = logging.getLogger(__name__)


class AmbiegenVAEGenerator:
    def __init__(self, executor=None, map_size=None):
        self.map_size = map_size
        self.beamng_executor = executor

    def initialize_parameters(self):
        log.info("Starting test generation, initializing parameters")
        self.tc_stats = {}
        self.tcs = {}
        self.tcs_convergence = {}

        self.seed = get_random_seed()
        time_budget = self.beamng_executor.time_budget.time_budget
        self.pop_size = min(int(round(time_budget/180)), 40)
        log.info(f"Population size: {self.pop_size}, time budget: {time_budget}")
        #self.num_gen = int(round(time_budget / 360)) 
        self.alg = "ga"
        self.sampl = "random"  
        self.crossover = "sbx"
        self.mutation = "pm"
        self.test_exec = "beam_exec"

    def initialize_vae(self):
        self.nDim = 17 
        self.nLat = 17
        self.archive = np.load("ambiegenvae\\vae_dataset.npy")

        mean = np.mean(self.archive, axis=0)  # mean for each feature
        std = np.std(self.archive, axis=0)
        self.transform = Denormalize1D(mean, std)
        self.model = VecVAE(self.nDim, self.nLat)
        self.model = load_model(1500, self.model, path="ambiegenvae\\VAE_models")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def initialize_problem(self):

        validator = RoadValidator(self.map_size)
        gen = KappaRoadGenerator(self.map_size, solution_size=self.nDim)
        self.latent_generator = LatentGenerator(
            self.nLat, 0, 1, gen, self.model, self.transform
        )
        executor = BeamExecutor(
            self.beamng_executor, self.latent_generator, test_validator=validator
        )
        min_fitness = self.beamng_executor.oob_tolerance
        self.problem = LKASVAEProblem(
            executor=executor,
            n_var=self.nLat,
            min_fitness=min_fitness,
            vae=self.model,
            transform=self.transform,
        )  

    def configure_algorithm(self):
        self.method = ALGORITHMS[self.alg](
            pop_size=self.pop_size,
            n_offsprings=int(round(self.pop_size / 2)),
            sampling=SAMPLERS[self.sampl](), 
            crossover=CROSSOVERS[self.crossover](prob=0.5, eta=3.0, vtype=float),
            mutation=MUTATIONS[self.mutation](prob=0.4, eta=3.0, vtype=float),
            eliminate_duplicates=AbstractDuplicateElimination(
                generator=self.latent_generator, threshold=0.13
            )
        )

    def run_optimization(self):
        res = minimize(
            self.problem,
            self.method,
            termination= BeamNGTermination(),
            seed=self.seed,
            verbose=False,
            eliminate_duplicates=True,
            save_history=False,
        )
        return res

    def save_results(self, res):
        self.tc_stats["run" + str(self.run)] = get_stats(res)
        self.tcs_convergence["run" + str(self.run)] = get_convergence(res)
        self.tcs["run" + str(self.run)] = get_test_suite(res)
        save_tc_results(
            self.tc_stats,
            self.tcs,
            self.tcs_convergence,
            "stats",
            self.alg,
            self.test_exec,
             "_" + self.sampl,
        )

    def start(self, run=0):
        self.run = run
        self.initialize_parameters()
        self.initialize_vae()
        self.initialize_problem()
        self.configure_algorithm()
        res = self.run_optimization()
        #self.save_results(res)
