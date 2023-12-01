
import logging as log
from time import sleep
import aren.src.utils as utils
import aren.src.debug as debug

from code_pipeline.test_analysis import compute_all_features
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.validation import TestValidator

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.termination import Termination
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer, Choice, Binary
import math
import random

class ArenGenerator():
    """
        Yassou
    """


    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):

        # # testing
        # points = [
        #     (0, 0),
        #     (10, 0),
        #     (20, 0),
        #     (10, 10),
        #     (0, 0)
        # ]

        # print(utils.heu_approxSelfIntersecting(points))
        # exit()

        # (1) GET PARAMETERS
        max_num_points = 30

        # (2) DEFINE THE FITNESS FUNCTIONS and CONSTRAINTS
        num_objectives = 3
        def get_heuristics(x):
            # x = [theta_p1, theta_p2, theta_p3, ...]

            heuristics = []

            # #################
            # STATIC VALIDATION
            # #################
            
            # HV0: adequate number of points, minimum length
            # handled by num_points bounds

            # Derive road points from angles
            road_points, missing_dist = utils.getRoadPointsFromAngles(x, self.map_size)

            # HV1: inside map
            # NOTE this is an approximation. if true, must fail
            hv1 = missing_dist

            assert (road_points==None and hv1 > 0) or (road_points!=None and hv1 == 0), f"p={road_points}, hv1={hv1}"

            # ESCAPE EARLY
            if hv1 > 0:
                return [hv1, float('inf'), float('inf')]
            
            # HV2: not self-intersecting
            # NOTE this is an approximation. if true, must fail
            hv2 = utils.heu_approxSelfIntersecting(road_points)

            if hv2 > 0:
                return [hv1, hv2, float('inf')]

            # Create the test
            the_test = RoadTestFactory.create_road_test(road_points)
            
            # HV3: too_sharp_turn:
            hv3 = utils.heu_tooSharpTurns(the_test)

            # ###############
            # STATIC FEATURES
            # ###############


            # TODO maybe here we can directly use
            features = compute_all_features(the_test, [])

            # HF1: dir_cov
            dir_cov = utils.getDirCov(x)



            # HF2: max_curv
            max_curv = utils.getMaxCurv(x)


            # ##########
            # SIMULATION
            # ##########
            # TODO only do this if hv1, hv2, hv3 are all 0s


            # result = [f"{hv1:.3f}", f"{hv2:.3f}", f"{hv3:.3f}"]
            # print(result)
            # is_valid = debug.validate(self.executor, the_test)
            # debug.visualise(self.executor, the_test)





            # ##################
            # DYNAMIC VALIDATION
            # ##################



            # # ### FITNESS (we want diversity of ...)

            # # DIR_COV:
            # x = utils.getDirCov(x)

            # # MAX_CURV:
            # X = utils.getMaxCurv(x)
            

            # # StdSA


            # # MLP


            # # StdSpeed


            # # TODO Run the scenario

            # # Scenario validity


            # # min distance to oob
            
            # # FITNESS = diversity of ... (????????) might be complex...
            # # FITNESS = runtime.oob value or percentage
            # # FITNESS = runtime.outcome (pass, fail, error, invalid)
            # # FITNESS = runtime.result
            # # FITNESS = runtime.when does the most dangerous moment occur? After that, the additional points become less and less relevant

            # # More: woudl need to define some kind of method to derive start point AFTER the path orientations have been derived

            # # TODO decide how to decide which pop member tosiumulate, which to not

            # # UPDATE: no need to start from point arund the region
            # # we can start from any point, as long as it fits

            
            # # HANDLE CONSTRAINT CATEGORIZATION
            # # con2id, exp = handleConstraints(scenario, constraints)



            return [hv1, hv2, hv3]


        # (3) DEFINE THE PROBLEM

        class SBFTProblemMixed(ElementwiseProblem):
            
            def __init__(self, **kwargs):
                vars = {}
                for i in range(max_num_points):
                    vars[f'p{i}_theta'] = Real(bounds=(-35, 35))
                vars[f'num_points'] = Integer(bounds=(2, max_num_points))
                super().__init__(vars=vars, n_obj=num_objectives, **kwargs)

            # Notes: x = [theta_p1, d_p1, theta_p2, d_p2, theta_p3, d_p3, ...]
            # TODO potential simplification: use a fixed d for all points
            def _evaluate(self, x, out, *args, **kwargs):
                heuristics = get_heuristics(x)
                out["F"] = heuristics
                
        problem =  SBFTProblemMixed()
    
        # (4) GET THE ALGORITHM
        # TODO try different algos
        # TODO try different parameters
        # TODO think about using the Scenic implementation of NSGA2, which includes restarts
        # algorithm = NSGA2(pop_size=100, n_offsprings=100, eliminate_duplicates=True)

        
        # TODO find a way to ensure that an executed test does not get into the next population.
        # instead, a failing test should be slightly modifies and re-added to the population
        algorithm = MixedVariableGA(pop_size=10, n_offsprings=10, survival=RankAndCrowdingSurvival())

        # (5) GET THE TERMINATION CRITERIA

        class SBFTTermination(Termination):

            def __init__(self, executor) -> None:
                super().__init__()
                self.executor = executor

            def _update(self, algorithm):
                return 1 if self.executor.time_budget.is_over() else 0

        termination = SBFTTermination(self.executor)

        # (6) RUN THE ALGORITHM
        # TODO look into using a seed
        print("Running the algorithm")

        
        seed = random.randint(0, 1000)
        seed=1565
        print(seed)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=1)






        # exit()




        # REQUIRE:
        # OPTIMIZE:
        # shorter tests


        # ### STEP 2:
        # use some simple search (ex binary-search) to determine a starting point.
        # technically, the starting point should not affect the result of simulation

        # REQUIRE:
        # OPTIMIZE:
        # diversity of direction

        # ### STEP 3:
        # run the simulation
        
        # RETURN:
        # relevant measurments (oob, oob_location, oob_percentage, etc)
        # feed the measurments to the NSGA to decide what to change in the orientations
        # feed the measurments to the simple search to help diversity measures for future iterations




        # POSSIBLE APPROACH 1:
        # have some kind of feedback loop
        # gen test, run, get features, get outcome
        # if outcome is bad, gen new test close to the previous one
        # if outcome is good, gen new test far from the previous one (to increase diversity)

        # I might be able to use the partial feature measurement (pre-xecution) as a guidance metric for optimization


        # POSSIBLE APPROACH 2:



        # END POSSIBLE APPROACHES

        road_points = []

        # IMPORTANT:
        # Prioritize shorter tests, to minimize execution time, which most time-consuming part

        # STEP 2
        the_test = RoadTestFactory.create_road_test(road_points)

        # STEP 3
        test_outcome, description, execution_data = self.executor.execute_test(the_test)

        

        # STEP 4: tentatively print some relevant info
        

        # Print the result from the test and continue
        oob_percentage = [state.oob_percentage for state in execution_data]
        log.info("Collected %d states information. Max is %.3f", len(oob_percentage), max(oob_percentage))

        log.info(f"Test_outcome: {test_outcome}")
        log.info(f"Description {description}")
        log.info(f"Data {execution_data}")


        # if self.executor.road_visualizer:
        #     sleep(5)
