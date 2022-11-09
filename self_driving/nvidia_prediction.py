import numpy as np

from self_driving.simulation_data import SimulationDataRecord
from self_driving.image_processing import preprocess

import tensorflow as tf

MIN_SPEED = 20


class NvidiaPrediction:
    def __init__(self, model, max_speed):
        self.model = model
        self.speed_limit = max_speed
        self.max_speed = max_speed

    def predict(self, image, car_state: SimulationDataRecord, normalize: bool = False):
        try:
            image = np.asarray(image)

            image = preprocess(image=image, normalize=normalize)
            image = np.array([image])

            with tf.device('/cpu:0'):
                steering_angle = float(self.model.predict(image, batch_size=1))

            speed = car_state.vel_kmh
            if speed > self.speed_limit:
                self.speed_limit = MIN_SPEED  # slow down
            else:
                self.speed_limit = self.max_speed

            throttle = np.clip(a=1.0 - steering_angle ** 2 - (speed / self.speed_limit) ** 2, a_min=0.0, a_max=1.0)
            return steering_angle, throttle

        except Exception as e:
            print(e)
