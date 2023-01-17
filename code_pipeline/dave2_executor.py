import numpy as np
from tensorflow.python.keras.models import load_model

from code_pipeline.executors import AbstractTestExecutor

import time
import traceback
from typing import Tuple

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_car_cameras import BeamNGCarCameras
# maps is a global variable in the module, which is initialized to Maps()
from self_driving.beamng_tig_maps import maps, LevelsFolder
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.nvidia_prediction import NvidiaPrediction
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader

from shapely.geometry import Point

import logging as log
import os.path

FloatDTuple = Tuple[float, float, float, float]


class Dave2Executor(AbstractTestExecutor):

    def __init__(self, result_folder, map_size, dave2_model,
                 time_budget=None,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None, debug=False):
        super(Dave2Executor, self).__init__(result_folder, map_size,
                                            time_budget=time_budget, debug=debug)

        self.risk_value = 0.7

        self.oob_tolerance = oob_tolerance
        self.maxspeed = max_speed

        self.brewer: BeamNGBrewer = None
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.model_file = dave2_model

        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')
        self.model = None

        # TODO Add checks with default setup. This requires a call to BeamNGpy resolve  (~/Documents/BeamNG.research)
        if self.beamng_user is not None and not os.path.exists(os.path.join(self.beamng_user, "research.key")):
            log.warning("%s is missing but is required to use BeamNG.research", )

        # Runtime Monitor about relative movement of the car
        self.last_observation = None
        # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
        self.min_delta_position = 1.0
        self.road_visualizer = road_visualizer

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        # TODO Show name of the test?
        log.info("Executing the test")

        # TODO Not sure why we need to repeat this 2 times...
        counter = 2

        attempt = 0
        sim = None
        condition = True
        while condition:
            attempt += 1
            if attempt == counter:
                test_outcome = "ERROR"
                description = 'Exhausted attempts'
                break
            if attempt > 1:
                self._close()
            if attempt > 2:
                time.sleep(5)

            sim = self._run_simulation(the_test)

            if sim.info.success:
                if sim.exception_str:
                    test_outcome = "FAIL"
                    description = sim.exception_str
                else:
                    test_outcome = "PASS"
                    description = 'Successful test'
                condition = False

        execution_data = sim.states

        # TODO: report all test outcomes
        return test_outcome, description, execution_data

    def _is_the_car_moving(self, last_state):
        """ Check if the car moved in the past 10 seconds """

        # Has the position changed
        if self.last_observation is None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if Point(self.last_observation.pos[0], self.last_observation.pos[1]).distance(
                Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position:
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def _run_simulation(self, the_test) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer(beamng_home=self.beamng_home, beamng_user=self.beamng_user)
            self.vehicle = self.brewer.setup_vehicle()

        # For the execution we need the interpolated points
        nodes = the_test.interpolated_points

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))

        # Override default configuration passed via ENV or hardcoded
        if self.beamng_user is not None:
            # Note This changed since BeamNG.research
            beamng_levels = LevelsFolder(os.path.join(self.beamng_user, '0.26', 'levels'))
            maps.beamng_levels = beamng_levels
            maps.beamng_map = maps.beamng_levels.get_map('tig')

        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = 'beamng_executor/sim_$(id)'.replace('$(id)', simulation_id)
        self.sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                          vehicle_state_reader=vehicle_state_reader,
                                                          simulation_name=name)

        # TODO: Hacky - Not sure what's the best way to set this...
        self.sim_data_collector.oob_monitor.tolerance = self.oob_tolerance
        self.sim_data_collector.get_simulation_data().start()

        try:
            brewer.bring_up()

            cameras = BeamNGCarCameras(beamng=beamng, vehicle=self.vehicle)

            if not self.model:
                self.model = load_model(self.model_file)
            predict = NvidiaPrediction(model=self.model, max_speed=self.maxspeed)

            while True:

                self.sim_data_collector.collect_current_data(oob_bb=True)
                last_state: SimulationDataRecord = self.sim_data_collector.states[-1]
                # Target point reached
                if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                    break

                assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(
                    self.sim_data_collector.name)

                assert not last_state.is_oob, "Car drove out of the lane " + str(self.sim_data_collector.name)

                data_img = cameras.cameras_array['cam_center'].poll()
                img = np.asarray(data_img['colour'].convert('RGB'))

                steering_angle, throttle = predict.predict(image=img, car_state=last_state, normalize=True)
                self.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)

                beamng.step(steps, wait=False)

            self.sim_data_collector.get_simulation_data().end(success=True)
        except AssertionError as aex:
            self.sim_data_collector.save()
            # An assertion that trigger is still a successful test execution, otherwise it will count as ERROR
            self.sim_data_collector.get_simulation_data().end(success=True, exception=aex)
            traceback.print_exception(type(aex), aex, aex.__traceback__)
        except Exception as ex:
            self.sim_data_collector.save()
            self.sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            self.sim_data_collector.save()
            try:
                self.sim_data_collector.take_car_picture_if_needed()
            except:
                pass

            self._close()

        return self.sim_data_collector.simulation_data

    def end_iteration(self):
        try:
            if self.brewer:
                self.brewer.beamng.stop_scenario()
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def _close(self):
        if self.brewer:
            try:
                # replicating beamng close method. The issue is that the kill_beamng method is not called for some
                # reason
                if self.brewer.beamng.scenario:
                    self.brewer.scenario.close()
                    self.brewer.scenario = None
                self.brewer.beamng.kill_beamng()

                time.sleep(2)

            except Exception as ex:
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None
