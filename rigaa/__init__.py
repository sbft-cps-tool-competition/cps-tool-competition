from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch

ALRGORITHMS = {
    "ga": GA,
    "nsga2": NSGA2,
    "rigaa": NSGA2,
    "random": RandomSearch,
}

