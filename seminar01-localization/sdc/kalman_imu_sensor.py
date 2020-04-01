# -*- coding: utf-8 -*-
import numpy as np
from .kalman_sensor_base import KalmanSensorBase


class KalmanImuSensor(KalmanSensorBase):
    """Калмановский IMU-датчик."""
    def __init__(self, *args, **kwargs):
        super(KalmanImuSensor, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'KalmanCAN'

    @property
    def observation_size(self):
        return 1
    
    def get_observation_matrix(self):
        observation_matrix = np.zeros(
            (self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._car_model.OMEGA_INDEX] = 1
        return observation_matrix
