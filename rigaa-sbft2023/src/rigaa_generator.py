import argparse
import sys
import logging as log

from pymoo.optimize import minimize
from pymoo.termination import get_termination

import src.config as cf
from src.rigaa import ALRGORITHMS
from src.rigaa.duplicate_elimination.duplicate_rem import DuplicateElimination
from src.rigaa.problems import PROBLEMS
from src.rigaa.samplers import SAMPLERS
from src.rigaa.search_operators import OPERATORS
from src.rigaa.utils.random_seed import get_random_seed
from src.rigaa.utils.callback import DebugCallback


class RIGAATestGenerator:
    def __init__(self, executor=None, map_size=None):
        self.map_size = map_size
        self.executor = executor
        # global my_executor
        # my_executor = executor

    def start(self, problem="vehicle", algo="rigaa", runs_number=1, random_seed=None, n_eval=None, full=True):
        """
        Function for running the optimization and saving the results"""

        while not self.executor.is_over():
            log.info("Running the optimization")
            log.info("Problem: %s, Algorithm: %s", problem, algo)

            if cf.ga["pop_size"] < cf.ga["test_suite_size"]:
                log.error("Population size should be greater or equal to test suite size")
                sys.exit(1)

            n_offsprings = int(cf.ga["pop_size"] / 3)
            if algo == "rigaa":
                rl_pop_percent = cf.rl["init_pop_prob"]
            else:
                rl_pop_percent = 0
            algorithm = ALRGORITHMS[algo](
                n_offsprings=n_offsprings,
                pop_size=cf.ga["pop_size"],
                sampling=SAMPLERS[problem](rl_pop_percent, self.map_size, self.executor),
                crossover=OPERATORS[problem + "_crossover"](cf.ga["cross_rate"], self.map_size, self.executor),
                mutation=OPERATORS[problem + "_mutation"](cf.ga["mut_rate"]),
                eliminate_duplicates=DuplicateElimination(),
                n_points_per_iteration=n_offsprings
            )

            if n_eval is None:
                termination = get_termination("n_gen", cf.ga["n_gen"])
                log.info("The search will be terminated after %d generations", cf.ga["n_gen"])
            else:
                termination = get_termination("n_eval", n_eval)
                log.info("The search will be terminated after %d evaluations", n_eval)

            if (random_seed is not None):
                seed = random_seed
            else:
                seed = get_random_seed()

            log.info("Using random seed: %s", seed)

            res = minimize(
                PROBLEMS[problem + "_" + algo](full=full),
                algorithm,
                termination,
                seed=seed,
                verbose=False,
                save_history=True,
                eliminate_duplicates=True,
                callback=DebugCallback()
            )

            log.info("Finished running the search.")
            log.info("Execution time, %f sec", res.exec_time)
            log.info(
                "Remaining time %s",
                self.executor.get_remaining_time(),
            )
            my_executor = res.algorithm.callback.executor
            self.executor.stats = my_executor.stats
