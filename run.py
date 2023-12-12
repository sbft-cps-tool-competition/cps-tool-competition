import os, sys, shutil
from pathlib import Path
import subprocess
import numpy as np
import datetime

python_exe = "python"
#python_exe = "C:\\Users\\japeltom\\PycharmProjects\\sbsf23\\venv2\\Scripts\\python.exe"

# RIGAA not listed as it needs Python 3.9
tools = ["crag", "evombt", "rigaa", "roadsign", "spirale", "wogan"]
tools = ["crag24"]
#tools = ["rigaa"]
commands = {
    "crag": "--module-path crag-sbft2023 --module-name src.crag --class-name CRAG ",
    "crag24": "--module-path crag-sbft2024 --module-name src.crag --class-name CRAG ",
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

    beamng_home = "C:/Users/stefan/Downloads/BeamNG.tech.v0.26.2.0"
    beamng_user = "C:/Users/stefan/BeamNG.tech"
    minutes = 60
    hours = 60 * minutes
    budget = 10 * minutes

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

def backup_simulations(sims_before, tool, dave2):
    sims_now = Path("./simulations/beamng_executor").glob("sim_2023*")
    new_sims = [sim for sim in sims_now if sim not in sims_before]

    print(len(new_sims), "New simulation directories")

    suffix = "dave2" if dave2 else "beamng"
    tool_sim_dir = f"./simulations/{tool}_{suffix}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(tool_sim_dir, exist_ok=True)
    for sim in new_sims:
        shutil.move(str(sim), tool_sim_dir)

# Clear simulation files from simulations/. Notice that you need to uncomment lines above.
clear_simulations()

# Randomize the tool execution order.
order = []
while min(order.count(tool) for tool in tools) < N:
    tool = np.random.choice(tools)
    if order.count(tool) < N:
        order.append(tool)

for tool in order:
    sims_before = list(Path("./simulations/beamng_executor").glob("sim_2023*"))
    run_on_powershell(python_exe, tool, dave2)
    backup_simulations(sims_before, tool, dave2)
