from rigaa.problems.vehicle_problem import VehicleProblem1Obj, VehicleProblem2Obj


PROBLEMS = {
    "vehicle_ga": VehicleProblem1Obj,
    "vehicle_nsga2": VehicleProblem2Obj,
    "vehicle_random": VehicleProblem1Obj,
    "vehicle_rigaa": VehicleProblem2Obj
}
