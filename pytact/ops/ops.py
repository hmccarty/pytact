#!/usr/bin/env python3

from abc import ABC
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rospy import Message as ROSMsg
from sensor_msgs.msg import Image
from typing import Dict, Tuple, Any, Optional

from .sensor import Sensor
from .util import *

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
