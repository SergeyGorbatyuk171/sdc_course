# -*- coding: utf-8 -*-
import numpy as np
from .car_sensor_base import CarSensorBase


class ImuSensor(CarSensorBase):
    """IMU-датчик. Измеряет угловую скорость автомобиля"""
    def __init__(self, *args, **kwargs):
        super(ImuSensor, self).__init__(*args, **kwargs)
        
    def __str__(self):
        return 'IMU'
    
    @property
    def observation_size(self):
        return 1
    
    def _observe_clear(self):
        return np.array([self._car._omega])
