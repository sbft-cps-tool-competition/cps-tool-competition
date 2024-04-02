
from rigaa.samplers.vehicle_sampling import VehicleSampling


from rigaa.rl_agents.vehicle_agent2 import generate_rl_road
from rigaa.samplers.vehicle_sampling import generate_random_road

SAMPLERS = {
    "vehicle": VehicleSampling,

}

GENERATORS ={
    "vehicle": generate_random_road,
    "vehicle_rl": generate_rl_road
}