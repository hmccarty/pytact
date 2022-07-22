#!/usr/bin/env python3

from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional

# from pytact.sensor import Sensor

class TactOp(ABC):
    """
    An operation to be performed on a tactile sensor.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def __call__(self):
        pass
