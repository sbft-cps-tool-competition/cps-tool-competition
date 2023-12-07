"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for saving test scenario images
"""

import os
import logging #as log
from datetime import datetime
from ambiegenvae.generators.abstract_generator import AbstractGenerator
log = logging.getLogger(__name__)

def save_tcs_images(generator: AbstractGenerator, test_suite, path, algo, problem, run, name, kappas=True):
    """
    It takes a test suite, a problem, and a run number, and then it saves the images of the test suite
    in the images folder

    Args:
      test_suite: a dictionary of solutions, where the key is the solution number and the value is the
    solution itself
      problem: the problem to be solved. Can be "robot" or "vehicle"
      run: the number of the runs
        algo: the algorithm used to generate the test suite. Can be "random", "ga", "nsga2",
    """

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    images_path = dt_string + "_" + path + "_" + algo + "_" + problem + "_" + name

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(os.path.join(images_path, "run" + str(run))):
        os.makedirs(os.path.join(images_path, "run" + str(run)))

    for i in range(len(test_suite)):

        path = os.path.join(images_path, "run" + str(run))

        test = test_suite[str(i)]
        if kappas:
            test = generator.genotype2phenotype(test)
    
        generator.visualize_test(test, path, i)

    log.info("Images saved in %s", images_path)
