
import optangle.src.utils as utils
from code_pipeline.tests_generation import RoadTestFactory

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer
from pymoo.core.termination import Termination
from pymoo.optimize import minimize

class OptAngleGenerator():
    """
        Generates test cases using a GA over a mixed-variable representation of the road.
        The road is defined by a sequence of road angles at a fixed distance from each other.
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):

        # (0) Keep a map of seen feature values to promote diversity

        # TODO future work:
        # Get better (dynamic?) bounds for the last 3 features 
        all_distributions = {}
        all_distributions['DIR_COV'] = utils.FeatureDistribution(0, 0.5)
        all_distributions['MAX_CURV'] = utils.FeatureDistribution(0, utils.TSHD_RADIUS)
        all_distributions['STD_SA'] = utils.FeatureDistribution(-180, 180)
        all_distributions['MEAN_LP'] = utils.FeatureDistribution(-2, 6)
        all_distributions['MAX_LP'] = utils.FeatureDistribution(-2, 6)
        
        NUM_OBJECTIVES = 3
        def get_heuristics(x):
            # x = [theta_p1, theta_p2, theta_p3, ...]

            # #################
            # STATIC VALIDATION
            # #################
            
            # HV0: adequate number of points, minimum length
            # handled by num_points bounds

            # Derive road points from angles
            road_points, missing_dist = utils.getRoadPointsFromAngles(x, self.map_size)

            # HV1: inside map
            # NOTE this is an approximation. if true, must fail.
            hv1 = utils.heu_missing_distance(missing_dist, self.map_size)

            assert (road_points==None and hv1 > 0) or (road_points!=None and hv1 == 0), f"p={road_points}, hv1={hv1}"

            # ESCAPE EARLY (1 and 1 are max(ish) values for hv2 and hv3) 
            if hv1 > 0:
                return [hv1+1+1, float('inf'), float('inf')]
            
            # HV2: not self-intersecting
            # NOTE this is an approximation. if true, must fail
            hv2 = utils.heu_approxSelfIntersecting(road_points)

            # ESCAPE EARLY (1 is max-ish value for hv3)
            if hv2 > 0:
                return [hv1+hv2+1, float('inf'), float('inf')]

            # Create the test
            the_test = RoadTestFactory.create_road_test(road_points)
            
            # HV3: too_sharp_turn:
            hv3 = utils.heu_tooSharpTurns(the_test)

            # ###############
            # STATIC FEATURES
            # ###############

            # TODO future work:
            # (?) look into using the static feature values to determine wether or not
            # to run the simulation
            # NOTE that computing the features at this stage is quite time-consuming

            # features = compute_all_features(the_test, [])
            # # HF1: dir_cov (how much it increases the diversity)
            # hf1, dc_score = utils.heu_diversity('DIR_COV', features, all_distributions)

            # # HF2: max_curv (how much it increases the diversity)
            # hf2, mc_score = utils.heu_diversity('MAX_CURV', features, all_distributions)       

            # TODO future work:            
            # (?) can we check whether the current test is close to a test that previously failed?
            # we could use this info to to determine wether or not to run the simulation

            # NOTE Current assumption: This is not needed
            # GA will eventually start to only suggest tests that are close to previously failed tests
            # so first, we explore, then we automatically start to exploit

            # ##########
            # SIMULATION
            # NOTE: in all cases, we want the test to FAIL, otherwise, the test is not interesting
            # ##########

            # Do we want to simulate?            
            # TODO future work
            # improve this to better control the exploration-exploitation trade-off (?)
            do_simulation = hv1 == 0 and hv2 == 0 and hv3 == 0

            # Initialize execution heuristics
            he0, he1, he2, he3, he4, he5 = float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')

            if do_simulation:

                # Test is valid, so we can simulate it
                test_outcome, description, execution_data = self.executor.execute_test(the_test)                
                
                if not the_test.is_valid:
                    # if the test is INVALID, we should update the hv# heuristics

                    # >>> Things that should never happen <<<
                    # "Wrong type"
                    # "Not enough road points."
                    # "The road definition contains too many points"
                    # "The road is not long enough."
                    
                    if description == "Not entirely inside the map boundaries":
                        # not expected to get in here at all
                        hv1 = 1
                    elif description == "The road is self-intersecting":
                        # This might happen, in the case where the centerline of the road
                        # is not self-intersecting, but the lanes of the road are.
                        hv2 = 0.1
                    elif description == "The road is too sharp":
                        # this is not expected, but might happen if TSHD_RADIUS is changed during evaluation
                        hv3 = 0.5
                    else:
                        # not expected to happen at all
                        hv1, hv2, hv3 = 1, 0.1, 0.5

                else:
                    # if the test is VALID, we evaluate the results.

                    # ##################
                    # EXECUTION ANALYSIS
                    # ##################

                    # HE0: OOB analysis (most important one, as this determines if the test fails or not)
                    # NOTE: no work done regarding the "test_outcome" value, as it is embeded in "oob_percentage"
                    oob_percentages = [state.oob_percentage for state in execution_data]
                    max_oob_percentage = max(oob_percentages)
                    he0 = 1 - max_oob_percentage

                    # TODO future work:
                    # when does the most dangerous moment occur? After that, the additional points become less and less relevant

                    # TODO future work:
                    # Add a metric to analyse the variance between types of OOBs (see the OOBAnalyzer class)

                    # HE1-5: Features diversity (only do this if the test fails)
                    if test_outcome == "FAIL":

                        features = the_test.features

                        # HE1: dir_cov (how much it increases the diversity)
                        he1, _ = utils.heu_and_add_diversity('DIR_COV', features, all_distributions)

                        # HE2: max_curv (how much it increases the diversity)
                        he2, _ = utils.heu_and_add_diversity('MAX_CURV', features, all_distributions)

                        # HE3: std_sa (how much it increases the diversity)
                        he3, _ = utils.heu_and_add_diversity('STD_SA', features, all_distributions)

                        # HE4: mean_lp (how much it increases the diversity)
                        he4, _ = utils.heu_and_add_diversity('MEAN_LP', features, all_distributions)

                        # HE5: max_lp (how much it increases the diversity)
                        he5, _ = utils.heu_and_add_diversity('MAX_LP', features, all_distributions)

                        # TODO future work:
                        # Keep a list of simuated tests (similar to the diversity lists) to avoid re-simulating them

            ###################
            # RETURN HEURISTICS
            ###################

            # TODO future work:
            # improve the weighting functions of the heuristics
            
            # pre-simulation validity heuristics (hv1, hv2, hv3: somewhat normalized)
            heu_validity = hv1+hv2+hv3
            
            # post-sim oob_percentage heuristic (he0: normalized)
            heu_oob_percentage = he0**3
            
            # post-sim (static and dynamic) features diversity heuristics (he1, he2, he3, he4, he5: normalized)
            heu_diversity = he1+he2+he3+he4+he5

            return [heu_validity, heu_oob_percentage, heu_diversity]


        # (1) DEFINE THE PROBLEM
        class SBFTProblemMixed(ElementwiseProblem):
            
            def __init__(self, **kwargs):
                vars = {}
                for i in range(utils.POINTS_RANGE[1]):
                    vars[f'p{i}_theta'] = Real(bounds=(-utils.THETA_MAX, utils.THETA_MAX))
                vars[f'num_points'] = Integer(bounds=utils.POINTS_RANGE)
                super().__init__(vars=vars, n_obj=NUM_OBJECTIVES, **kwargs)

            # Notes: x = [theta_p1, d_p1, theta_p2, d_p2, theta_p3, d_p3, ...]
            # TODO potential simplification: use a fixed d for all points
            def _evaluate(self, x, out, *args, **kwargs):
                heuristics = get_heuristics(x)
                out["F"] = heuristics
                
        problem =  SBFTProblemMixed()
    
        # (2) GET THE ALGORITHM
        # TODO future work:
        # fine-tune this (algorithm, parameters, Scenic implementation of NSGA2 w/ restarts)
        
        # TODO future work:
        # find a way to ensure that an executed test does not get into the next population.
        algorithm = MixedVariableGA(pop_size=10, n_offsprings=10, survival=RankAndCrowdingSurvival())

        # (3) GET THE TERMINATION CRITERIA
        class SBFTTermination(Termination):

            def __init__(self, executor) -> None:
                super().__init__()
                self.executor = executor

            def _update(self, algorithm):
                return 1 if self.executor.time_budget.is_over() else 0

        termination = SBFTTermination(self.executor)

        # (4) RUN THE ALGORITHM

        # TODO future work:
        # Think about using a seed
        res = minimize(problem, algorithm, termination, verbose=1)

