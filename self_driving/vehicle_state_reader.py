from collections import namedtuple
import numpy as np
from beamngpy import Vehicle, BeamNGpy
from beamngpy.sensors import Electrics, Timer, Sensor, State
from typing import List, Tuple

VehicleStateProperties = ['timer', 'pos', 'dir', 'vel', 'steering', 'steering_input',
                          'brake', 'brake_input', 'throttle', 'throttle_input',
                          'wheelspeed', 'vel_kmh']

VehicleState = namedtuple('VehicleState', VehicleStateProperties)


class VehicleStateReader:
    def __init__(self, vehicle: Vehicle, beamng: BeamNGpy):
        self.vehicle = vehicle

        self.beamng = beamng
        self.state: VehicleState = None
        self.vehicle_state = {}

        #assert 'state' in self.vehicle.sensors.keys(), "Default state sensor is missing"
        # Starting from BeamNG.tech 0.23.5_1 once the scenario is over a vehicle's sensors get automatically detached
        # Including the default state sensor so we need to ensure that is there somehow, or stop reusing the vehicle
        # object across simulations
        try:
            state = State()
            self.vehicle.attach_sensor('state', state)
        except:
            pass

        electrics = Electrics()
        timer = Timer()

        self.vehicle.attach_sensor('electrics', electrics)
        self.vehicle.attach_sensor('timer', timer)

    def get_state(self) -> VehicleState:
        return self.state

    def get_vehicle_bbox(self) -> dict:
        return self.vehicle.get_bbox()

    def update_state(self):
        self.vehicle.poll_sensors()

        st = self.vehicle.sensors['state']
        ele = self.vehicle.sensors['electrics']
        vel = tuple(st['vel'])

        self.state = VehicleState(timer=self.vehicle.sensors['timer'].data['time']
                                  , pos=tuple(st['pos'])
                                  , dir=tuple(st['dir'])
                                  , vel=vel
                                  , steering=ele.data.get('steering', None)
                                  , steering_input=ele.data.get('steering_input', None)
                                  , brake=ele.data.get('brake', None)
                                  , brake_input=ele.data.get('brake_input', None)
                                  , throttle=ele.data.get('throttle', None)
                                  , throttle_input=ele.data.get('throttle_input', None)
                                  , wheelspeed=ele.data.get('wheelspeed', None)
                                  , vel_kmh=int(round(np.linalg.norm(vel) * 3.6)))
