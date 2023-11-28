from base import RoadtitionBase

'''
Road generator:
    0)Iteratively: until the time budget is not over
     1)A initial population is generated, seen as a set of spiral arcs
     2)A heir population is generated ,joining the arcs with the Cartesian product between the inital set with itself ( excluding equal indices )
     3)A fitness value is associated to the heir population
     4)A new heir population is derived as a subset of the first one, and also valued with the fitness value
     5)The two fitness value are compared and IF the fitness of the first is minus of the new one than goes to the --> 1)
        6)else goes to the ----> 4)

    i.e: 
    1) initial population=[S1, S2, S3]
    2) heir population =[S1US2, S1US3, S2US1, S2US3, S3US1, S3US2] with fitness value = 6
    3) derived heir population =[S1US2, S2US3]  with fitness value = 2
    4) 6>2   --> new heir population =[(S1US2)U(S2US3), (S2US3)U(S1US2)] with fitness value = 3
    5) 2<3>  ----> 1) a new initial population is generated
'''

class Roadtition(RoadtitionBase):
    def __init__(self, executor=None, map_size=None):
        self.executor=executor
        super().__init__(executor=executor, map_size=map_size)

    def start(self):

        while self.executor.time_budget.get_remaining_real_time() > 0:

            #print("\nSono nella popolazione iniziale")
            initial_roads_population = self.initial_population_generator(n_max_of_road = 3)

            #print("\nComincio a fare il crossover")
            heir_population_1 , oob = self.hebi_generator(initial_roads_population)
            heir_population , fitness = self.fitness_of_one_population(heir_population_1 , oob, len(heir_population_1))
            #print("\nheir_population: ", heir_population)

            #print("\n--------------Entro nella fitness function-----------------------")
            self.fitness_generator(heir_population = heir_population ,fitness_parent = fitness, oob = oob)

        return None

