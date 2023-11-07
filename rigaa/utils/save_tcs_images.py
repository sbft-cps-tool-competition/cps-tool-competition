
import os
import logging as log

from rigaa.solutions.vehicle_solution import VehicleSolution
from datetime import datetime
import config as cf


def save_tcs_images(test_suite, problem, run, algo):
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

    images_path = dt_string + "_" + cf.files["images_path"] +  "_" + algo + "_" + problem

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(os.path.join(images_path, "run" + str(run))):
        os.makedirs(os.path.join(images_path, "run" + str(run)))

    for i in range(len(test_suite)):

        path = os.path.join(images_path, "run" + str(run), str(i) + ".png")

        if problem == "vehicle":
            VehicleSolution.build_image(test_suite[str(i)], path)
    log.info(
        "Images saved in %s", images_path
    )
