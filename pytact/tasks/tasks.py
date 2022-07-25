from abc import ABC, abstractmethod
from pytact.sensors import Sensor

class Task(ABC):
    """
    An operation to be performed on a tactile sensor.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, "_" + k, v)

    @abstractmethod
    def __call__(self, sensor: Sensor):
        pass
