from abc import ABC, abstractmethod
import cv2
import threading
from typing import Dict, Tuple, Any, Optional
from copy import deepcopy
import numpy as np

from .types import FrameEnc, Frame, Markers

class Sensor(ABC):

    """ Properties """
    @property
    def marker_shape(self) -> Optional[Tuple[int, int]]:
        return None

    @property                 
    def reference(self) -> Optional[Frame]:     
        return self._ref

    @reference.setter
    def reference(self, ref: Frame):
        self._ref = ref

    """ Required methods """
    @abstractmethod 
    def get_frame(self) -> Frame:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    """ Optional methods """
    def get_markers(self) -> Markers:
        raise NotImplementedError()

class GelsightR15(Sensor):
    """
    Retrieves images from an http stream and assumes markers are present.

    Parameters
    ----------
    url: str
        URL of the HTTP stream
    width: int, optional
        Downscaled width of image (default 120)
    height: int, optional
        Downscaled height of image (default 160)
    encoding: FrameEnc, optional
        Encoding to process frames as (default FrameEnc.BGR)
    sample_rate: float, optional
        Rate to capture new frames at (default 30.0)
    marker_shape: Tuple[int, int], optional
        Number of marker rows and columns (default (10, 12))
    """

    encoding = FrameEnc.BGR
    size: Tuple[int, int] = (120, 160) # width, height
    sample_rate: float = 30.0
    marker_shape: Tuple[int, int] = (10, 12) # rows, cols

    def __init__(self, url: str, **kwargs):
        self._dev = cv2.VideoCapture(url)
        self.size = (
            int(kwargs["width"]) if "width" in kwargs else self.size[0],
            int(kwargs["height"]) if "height" in kwargs else self.size[1],
        )

        self.output_coords = [(0, 0), (self.size[0], 0), self.size, (0, self.size[1])]
        self._roi = kwargs["roi"] if "roi" in kwargs else None

        self._is_running = True 
        self._frame = None
        threading.Timer(1.0/self.sample_rate, self.collect_frame).start()

    def collect_frame(self):
        """Runs at predetermined rate to collect frames from the sensor."""
        ret, frame = self._dev.read()
        if not ret:
            self. is_running = False
            return

        # Warp to match ROI
        if self._roi is not None:
            M = cv2.getPerspectiveTransform(
                np.float32(self._roi), np.float32(self.output_coords))
            frame = cv2.warpPerspective(frame, M, self.size)

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame = Frame(frame, self.encoding)

    def marker_shape(self) -> Tuple[int, int]:
        return self.marker_shape

    def get_frame(self) -> Optional[Frame]:
        """Returns frame collected in the last sample."""
        return deepcopy(self._frame)

    def is_running(self):
        return self._ret
        