from dataclasses import dataclass
from enum import Enum
import numpy as np

class FrameEnc(Enum):
    """Supported encodings for frames"""
    BGR = 1 # 3 channels in blue, green, red order
    GRAY = 2 # 1 channel

@dataclass
class Frame:
    """Raw tactile array"""
    encoding: FrameEnc
    image: np.ndarray # (height, width, 1-3)

    def clamp(self):
        """Limits image data by encoding"""

        if encoding == FrameEnc.BGR or encoding == FrameEnc.MONO:
            self.image[self.image > 255] = 255
            self.image[self.image < 0] = 0

@dataclass
class Markers:
    """Key areas of interest within a frame"""
    rows: int
    cols: int 
    markers: np.ndarray # (num_markers, 2)

    def __post_init__(self):
        if self.rows <= 0:
            raise ValueError(f"GelsightMarkers: rows cannot be less than or equal to 0, given: {self.rows}")
        elif self.cols <= 0:
            raise ValueError(f"GelsightMarkers: cols cannot be less than or equal to 0, given: {self.cols}")
        elif len(self.markers.shape) != 2 or self.markers.shape[1] != 2:
            raise ValueError(f"GelsightMarkers: markers must have shape (n_markers, 2), given shape: {self.markers.shape}")

@dataclass
class Flow:
    """Matched markers between a reference and current frame"""
    ref: Markers
    cur: Markers

    def __post_init__(self):
        if len(self.ref.markers) != len(self.cur.markers):
            raise ValueError(f"Reference and current markers have different sizes")

@dataclass
class DepthMap:
    """3d data of a frame"""
    data: np.ndarray # (height, width, dtype=float32)
