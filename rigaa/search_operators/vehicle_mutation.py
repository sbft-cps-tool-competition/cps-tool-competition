import copy
from pymoo.core.mutation import Mutation
import numpy as np
import config as cf

import logging as log

class VehicleMutation(Mutation):
    """
    Module to perform the mutation
    """

    def __init__(self, mut_rate):
        super().__init__()
        self.mut_rate = mut_rate

    def _do(self, problem, X, **kwargs):

        for i in range(len(X)):
            r = np.random.random()
            s = X[i, 0]
            # with a given probability - perform the mutation
            if r < self.mut_rate:

                log.debug("Mutation performed on individual %s", s)
                

                sn = copy.deepcopy(s)

                wr = np.random.random()
                child = copy.deepcopy(sn.states)
                # exchnage mutation operator, exchange two random states
                n = np.random.randint(1, 4)
                if wr < 0.5:
                    
                    while n > 0:
                        log.debug("Exchange mutation performed on individual %s", s)
                        candidates = list(np.random.randint(0, high=len(child)-1, size=2))
                        temp = child[candidates[0]].copy()
                        child[candidates[0]] = child[(candidates[1])]
                        child[(candidates[1])] = temp
                        n -= 1
                # change of value operator, change the value of one of the attributes of a random state
                else:#if wr < 0.9:
                    while n > 0:
                        log.debug("Change of value mutation performed on individual %s", s)
                        num = np.random.randint(0, high=len(child)-1)
                        if child[(num)][0] == 0:
                            child[(num)][0] = np.random.choice([1, 2])

                        elif child[(num)][0] == 1:
                            child[(num)][0] = np.random.choice([0, 2])

                        elif child[(num)][0] == 2:
                            child[(num)][0] = np.random.choice([0, 1])

                        if child[(num)][0] == 0:
                            value_list = np.arange(
                                cf.vehicle_env["min_len"], cf.vehicle_env["max_len"], 2
                            )
                            child[num][1] = int(np.random.choice(value_list))
                        else:
                            value_list = np.arange(
                                cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"], 5
                            )
                            child[num][2] = int(np.random.choice(value_list))
                        n -= 1

                sn.states = child.copy()
                X[i, 0] = sn

        return X