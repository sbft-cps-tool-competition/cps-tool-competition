import logging as log

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

from self_driving.decal_road import DecalRoad
from self_driving.road_points import List4DTuple, RoadPoints
from self_driving.simulation_data import SimulationParams
from self_driving.beamng_pose import BeamNGPose


class BeamNGCamera:
    def __init__(self, beamng: BeamNGpy, name: str, camera: Camera = None):
        self.name = name
        self.pose: BeamNGPose = BeamNGPose()
        self.camera = camera
        if not self.camera:
            self.camera = Camera((0, 0, 0), (0, 0, 0), 120, (1280, 1280), colour=True, depth=True, annotation=True)
        self.beamng = beamng

    def get_rgb_image(self):
        self.camera.pos = self.pose.pos
        self.camera.direction = self.pose.rot
        cam = self.beamng.render_cameras()
        img = cam[self.name]['colour'].convert('RGB')
        return img


class BeamNGBrewer:
    def __init__(self, beamng_home=None, beamng_user=None, road_nodes: List4DTuple = None):
        self.scenario = None

        # This is used to bring up each simulation without restarting the simulator
        self.beamng = BeamNGpy('localhost', 64256, home=beamng_home, user=beamng_user)
        self.beamng.open(launch=True)

        # We need to wait until this point otherwise the BeamNG logger level will be (re)configured by BeamNGpy
        log.info("Disabling BEAMNG logs")
        for id in ["beamngpy", "beamngpy.beamngpycommon", "beamngpy.BeamNGpy", "beamngpy.beamng", "beamngpy.Scenario",
                   "beamngpy.Vehicle", "beamngpy.Camera"]:
            logger = log.getLogger(id)
            logger.setLevel(log.CRITICAL)
            logger.disabled = True

        self.vehicle: Vehicle = None
        if road_nodes:
            self.setup_road_nodes(road_nodes)

        steps = 80
        self.params = SimulationParams(beamng_steps=steps, delay_msec=int(steps * 0.05 * 1000))
        self.vehicle_start_pose = BeamNGPose()

    def setup_road_nodes(self, road_nodes):
        self.road_nodes = road_nodes
        self.decal_road: DecalRoad = DecalRoad('street_1').add_4d_points(road_nodes)
        self.road_points = RoadPoints().add_middle_nodes(road_nodes)

    def setup_vehicle(self) -> Vehicle:
        assert self.vehicle is None
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='TIG', color='Red')
        return self.vehicle

    def bring_up(self):

        # After 1.18 to make a scenario one needs a running instance of BeamNG
        self.scenario = Scenario('tig', 'tigscenario')
        if self.vehicle:
            self.scenario.add_vehicle(self.vehicle, pos=self.vehicle_start_pose.pos,
                                      rot_quat=self.vehicle_start_pose.rot)

        self.scenario.make(self.beamng)
        self.beamng.set_deterministic()
        # self.beamng.set_steps_per_second(120)  # Set simulator to 60hz temporal resolution
        # self.beamng.remove_step_limit()
        self.beamng.load_scenario(self.scenario)

        self.beamng.start_scenario()

        self.beamng.pause()
