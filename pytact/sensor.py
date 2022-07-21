#!/usr/bin/env python3

from abc import ABC
import cv2
import time
import threading
from typing import Dict, Tuple, Any, Optional
from copy import deepcopy

from .data import FrameEnc, Frame

class Sensor(ABC):

    @property
    def marker_shape(self) -> Optional[Tuple[int, int]]:
        return None

    @property                 
    def reference(self) -> Optional[Frame]:     
        return self._ref

    @reference.setter
    def reference(self, ref: Frame):
        self._ref = ref

    @abstractmethod 
    def get_frame(self) -> Frame:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

class GelsightR15(Sensor):

    encoding = FrameEnc.BGR
    size: Tuple[int, int] = (120, 160) # width, height
    sample_rate: float = 30.0
    marker_shape: Tuple[int, int] = (10, 12) # rows, cols

    def __init__(self, cfg: Dict[str, Any]):
        if "url" not in cfg:
            raise RuntimeError("Missing stream url.")
        self._dev = cv2.VideoCapture(cfg["url"])
        self.size = (
            int(cfg["width"]) if "width" in cfg else self.size[0],
            int(cfg["height"]) if "height" in cfg else self.size[1],
        )

        self.output_coords = [(0, 0), (self.size[0], 0), self.size, (0, self.size[1])]
        self._roi = cfg["roi"] if "roi" in cfg else None

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
        return 

    def get_frame(self) -> Optional[Frame]:
        """Returns frame collected in the last sample."""
        return deepcopy(self._frame)

    def is_running(self):
        return self._ret
        