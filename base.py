import math
import numpy as np
from random import randint, random
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer

class RoadtitionBase():
    def __init__(self, executor=None, map_size=None):
        self.executor = executor  # attribute that defines the type of executor
        self.map_size = map_size  # attribute specifying the square size of the map
        self.visualize = RoadTestVisualizer(self.map_size)  # To display the roads on the screen

    def execute_test(self, road_points):
        '''
        Method used to generate tests

        :param road_points: list of road seen as cartesian points
        :return: the result of the test [Pass, Fail, Invalid or Error]
        '''

        # I turn the road into a necessary format for testing
        test = RoadTestFactory.create_road_test(road_points)
        # With execute_tes you test the previously created road
        test_outcome, description, execution_data = self.executor.execute_test(test)
        print("\033[1;34m test_outcome= \033[1;31m", test_outcome, "\033[1;30m")
        #print("\033[1;34m execution_data.oob= \033[1;31m", execution_data[0].oob_distance, "\033[1;30m")
        self.visualize.visualize_road_test(test)

        return test_outcome , execution_data

    def initial_population_generator(self, n_max_of_road):
        '''
        Generates the initial population

        :param n_max_of_road: max number of road generated for the starting population

        :return: list of road seen as initial population
        '''
        i = 0
        road_population  = []

        while i<n_max_of_road  :
            if (i % 2 == 0): #If i is even then I generate an arc of spiral in counterclockwise direction Sx <--
                road_points = self.bow(c_x0=randint(60, 120), c_y0=randint(60, 100), radius=randint(40, 70),
                                       interpolation_points=randint(4, 5), Angle_init=randint(295, 360),
                                       Angle_final=randint(180, 245))

            else: # If i is odd then I generate an arc of spiral in clockwise direction --> Dx
                road_points = self.bow(c_x0=randint(60, 120), c_y0=randint(60, 100), radius=randint(40, 70),
                                       interpolation_points=randint(4, 5), Angle_init=randint(0, 65),
                                       Angle_final=randint(115, 180))

            x, y = self.roadpoint_to_xy(road_points)
            x_reframe, y_reframe = self.reframe(x, y)
            road_point_refremed = self.xy_to_roadpoint(x_reframe, y_reframe)
            test_outcome , execution_data = self.execute_test(road_point_refremed)

            if test_outcome != "ERROR" and test_outcome != "INVALID":
                road_population .append(road_point_refremed)
                #if test_outcome != "INVALID":
                    #print("\033[1;34m execution_data.oob= \033[1;31m", execution_data[0].oob_distance, "\033[1;30m")
                    #IN BASE AL oob crea un vettore anche on gli oob a valore minore in execution_data e poi la funzione fittin la
                    #fai andando a prendere il valore più piccolo della somma delle prime 5 tipo

            i = i + 1

        return road_population

    def hebi_generator(self, road_population):

        heir_population=[]
        oob=[]

        for i in range(0, len(road_population)):
            for j in range(0, len(road_population)):

                if i != j:
                    x1, y1 = self.roadpoint_to_xy(road_population[i])  # I transform the first into x,y coordinates
                    x2, y2 = self.roadpoint_to_xy(road_population[j])  # I transform the second into x,y coordinates
                    x_union, y_union = self.crossover(x1, y1, x2, y2)  # crossover between the 2 roads
                    x_reframe, y_reframe = self.reframe(x_union, y_union)
                    new_road_point = self.xy_to_roadpoint(x_reframe, y_reframe)
                    test_outcome , execution_data = self.execute_test(new_road_point)
                    if test_outcome != "ERROR" and test_outcome != "INVALID" :
                        heir_population.append( new_road_point )
                        oob.append(execution_data[0].oob_distance)
                        print("\033[1;34m execution_data.oob= \033[1;31m", execution_data[0].oob_distance,"\033[1;30m")
        return heir_population ,oob

    def bow(self, c_x0, c_y0, radius, interpolation_points, Angle_init, Angle_final):
        '''
        Generator of roads modeled as spiral arcs

        :param c_x0:
        :param c_y0:
        :param radius:
        :param interpolation_points:
        :param Angle_init:
        :param Angle_final:
        :return:
        '''
        raggio_v = radius
        random_coef= randint(4,5)
        road_points = []

        center_x = c_x0
        center_y = c_y0

        angles_in_deg = np.linspace(Angle_init, Angle_final, num=interpolation_points)

        for angle_in_rads in [math.radians(a) for a in angles_in_deg]:
            raggio_v = raggio_v + random() * random_coef
            x = math.sin(angle_in_rads) * raggio_v + center_x
            y = math.cos(angle_in_rads) * raggio_v + center_y
            road_points.append((x, y))


        return road_points

    def crossover(self, x1, y1, x2, y2):
        '''
        crossover between the 2 roads

        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        '''
        x_temp = []
        x_diff = x2[0] - x1[-1]
        y_temp = []
        y_diff = y2[0] - y1[-1]
        for t in range(0, len(x2)):
            x_temp.append(x2[t] - x_diff)

        for i in range(0, len(y2)):
            y_temp.append(y2[i] - y_diff)

        del x_temp[0]
        del y_temp[0]

        #del x1[-1]
        #del y1[-1]

        x_unione = [*x1, *x_temp]
        y_unione = [*y1, *y_temp]

        return x_unione, y_unione

    def xy_to_roadpoint(self, x, y):
        '''
        From 2 list of points to a list of
        :param x:
        :param y:
        :return:
        '''
        road_point = []
        for i in range(0, len(x)):
            road_point.append((x[i], y[i]))
        return road_point

    def roadpoint_to_xy(self, road_point):
        '''

        :param road_point:
        :return:
        '''
        x = [x for x, y in road_point]
        y = [y for x, y in road_point]
        return x, y

    def reframe(self, x_unione, y_unione):
        '''

        :param x_unione:
        :param y_unione:
        :return:
        '''
        x_reframe = x_unione
        y_reframe = y_unione

        if (min(x_unione) < 10 or max(x_unione) > 180) and (max(x_unione) - min(x_unione)) < 180:
            if (min(x_unione) < 10):
                inc_x = 10 - min(x_unione)
                for i in range(0, len(x_unione)):
                    t = x_unione[i]
                    x_reframe[i] = t + inc_x
            elif (max(x_unione) > 180):
                dec_x = 180 - max(x_unione)
                for i in range(0, len(x_unione)):
                    t = x_unione[i]
                    x_reframe[i] = t + dec_x

        if (min(y_unione) < 10 or max(y_unione) > 180) and (max(y_unione) - min(y_unione)) < 180:
            if (min(y_unione) < 10):
                inc_y = 10 - min(y_unione)
                for i in range(0, len(y_unione)):
                    t = y_unione[i]
                    y_reframe[i] = t + inc_y
            elif (max(y_unione) > 180):
                dec_y = 180 - max(y_unione)
                for i in range(0, len(y_unione)):
                    t = y_unione[i]
                    y_reframe[i] = t + dec_y

        if (max(x_unione) - min(x_unione)) > 180 or (max(y_unione) - min(y_unione)) > 180:
            x_d, y_d = self.halve(x_reframe, y_reframe)
            x_reframe, y_reframe = self.reframe(x_d, y_d)

        return x_reframe, y_reframe

    def halve(self, x, y):
        '''

        :param x:
        :param y:
        :return:
        '''

        slice_x = len(x) // 4
        del x[0:slice_x]
        del x[len(x) - slice_x:len(x)]

        slice_y = len(y) // 4
        del y[0:slice_y]
        del y[len(y) - slice_y:len(y)]

        return x, y


    def fitness_generator(self,heir_population , fitness_parent, oob):

        v = True
        while fitness_parent >= 0 and v:

            if len(heir_population) > 4:
                new_population , new_fitness = self.fitness_of_one_population(heir_population, oob , 5) #mi estrapolo solo le 5 strade che hanno fitness più basse, se ci sono!
                #print("\nla nuova popolazione con oob minimo è:" , new_population," ed ha come funzione fitness:\033[1;31m",new_fitness,"\033[1;30m")

                new_heir_population , oob=self.hebi_generator(new_population) #le incrocio

                new_population[0:len(new_population)]=[] #azzero la popolazione per cacolarmene una nuova

                fitness_parent = fitness_parent - new_fitness  # se il fitness di prima è più grande di quella nuova ok, continuo perché voglio diminuirla

            else:
                v = False
                #print("Entro nell'else perché la popolazione non è abbastanza grande. Metto a False!")


        return None

    def fitness_of_one_population(self, heir_population , oob , n):

        new_population = []
        fitness = 0
        oob_temp=oob[:]
        for i in range(0, n):  # estraggo n valori dalla popolazione
            minimo = min(oob_temp) # mi caolcolo la strada con oob minimo
            fitness = fitness + minimo # faccio la sommatoria tra gli n minimi
            indice_minimo = oob_temp.index(minimo)# mi ricavo l'indice
            new_population.append(heir_population[indice_minimo]) # in modo da accodare la strada alla nuova popolazione con solo gli oob minori
            del heir_population[indice_minimo] # e poi li elimino per non riprendere le stesse strade
            del oob_temp[indice_minimo]

        return new_population , fitness
