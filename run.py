import os, sys, shutil
from pathlib import Path
import subprocess
import numpy as np

python_exe = "C:\\Users\\japeltom\\PycharmProjects\\sbsf23\\venv\\Scripts\\python.exe"

# RIGAA not listed as it needs Python 3.9
tools = ["crag", "evombt", "roadsign", "spirale", "wogan"]
commands = {
    "crag": "--module-path crag-sbft2023 --module-name src.crag --class-name CRAG ",
    "evombt": "--module-path evombt_generator --module-name evombt_generator --class-name EvoMBTGenerator ",
    "rigaa": "--module-path rigaa-sbft2023 --module-name src.rigaa_generator --class-name RIGAATestGenerator ",
    "roadsign": "--module-path roadsign-sbft2023 --module-name src.roadsign_generator --class-name RoadSignGenerator ",
    "spirale": "--module-path spirale-sbft2023 --module-name src.main --class-name Roadtition ",
    "wogan": "--module-path wogan-sbft2023 --module-name src.wogan --class-name WOGAN "
}

# How many times to run each tool.
try:
    N = int(sys.argv[1])
except:
    raise Exception("Please specify the number of times each tool should be executed.")

dave2 = len(sys.argv) > 2 and sys.argv[2].lower() == "dave2"

def run_on_powershell(python_exe, tool, dave2=False):
    python_exe = python_exe.strip()

    beamng_home = "C:/Users/japeltom/BeamNG/BeamNG.tech.v0.26.2.0"
    beamng_user = "C:/Users/japeltom/Documents/BeamNG.research"
    #budget = 3*3600
    budget = 150

    command = "{} competition.py --map-size 200 --beamng-home {} --beamng-user {} --time-budget {} ".format(python_exe, beamng_home, beamng_user, budget)
    command += commands[tool]

    if dave2:
        command += "--executor dave2 --dave2-model dave2/beamng-dave2-competition-strong.h5 --oob-tolerance 0.1 --speed-limit 25"
    else:
        command += "--executor beamng"

    p = subprocess.Popen(["powershell.exe", command], stdout=sys.stdout)
    p.communicate()

def clear_simulations():
    # This is destructive. Uncomment to remove the simulation files.
    #path = "simulations/beamng_executor"
    #if os.path.exists(path):
    #    shutil.rmtree(path)
    pass

# Clear simulation files from simulations/. Notice that you need to uncomment lines above.
clear_simulations()

# Randomize the tool execution order.
order = []
while min(order.count(tool) for tool in tools) < N:
    tool = np.random.choice(tools)
    if order.count(tool) < N:
        order.append(tool)

for tool in order:
    run_on_powershell(python_exe, tool, dave2)
