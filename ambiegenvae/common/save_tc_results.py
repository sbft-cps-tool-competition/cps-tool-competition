"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for saving test scenario metadata such as the test suite statistics and the test suite itself
"""
from datetime import datetime
import os
import json
import logging #as log
log = logging.getLogger(__name__)


def save_tc_results(tc_stats, tcs, tcs_convergence, path, algo, problem, name):
    """
    It takes two arguments, tc_stats and tcs, and saves them as JSON files in the directories specified
    in the config file

    Args:
      tc_stats: a dictionary of the test cases statistics
      tcs: a list of test cases
    """

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    stats_path = dt_string + "_" + path + "_" + algo + "_" + problem + "_" + name
    tcs_path = dt_string + "_" + path + "_" + algo + "_" + problem + "_" + name

    if not os.path.exists(stats_path):
        os.makedirs(stats_path, exist_ok=True)
    if not os.path.exists(tcs_path):
        os.makedirs(tcs_path, exist_ok=True)

    with open(
        os.path.join(stats_path, dt_string + "-stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tc_stats, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-stats.json")
        )

    with open(
        os.path.join(stats_path, dt_string + "-conv.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tcs_convergence, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-conv.json")
        )

    with open(
        os.path.join(tcs_path, dt_string + "-tcs.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tcs, f, indent=4)
        log.info(
            "Test cases saved as %s", os.path.join(tcs_path, dt_string + "-tcs.json")
        )
