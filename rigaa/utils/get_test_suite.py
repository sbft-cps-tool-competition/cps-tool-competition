
import config as cf
import logging as log
from rigaa.utils.car_road import Map
def get_test_suite(res, algo):
    """
    It takes the last generation of the population and returns a dictionary of 30 test cases

    Args:
      res: the result of the genetic algorithm

    Returns:
      A dictionary of 30 test cases.
    """
    test_suite = {}
    gen = len(res.history) - 1
    
    population = res.history[gen].pop.get("X")
    if algo != "nsga2" and algo != "rigaa":
        population = sorted(population, key=lambda x: abs(x[0].fitness), reverse=True)
    for i in range(cf.ga["test_suite_size"]):
        #result = res.history[gen].pop.get("X")[i][0]
        result = population[i][0]
        states = result.states
        test_map = Map(result.map_size)
        #car = Car(self.speed, self.steer_ang, self.map_size)
        road_points = test_map.get_points_from_states(states)
        '''
        new_states = []
        for state in states:
            new_states.append([int(x) for x in state])
        '''
        
        test_suite[str(i)] = road_points

    log.info("Test suite of %d test scenarios generated", cf.ga["test_suite_size"])
    return test_suite
