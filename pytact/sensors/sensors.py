from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

from pytact.types import ModelType, Frame, Markers

class UnsupportedModelError(Exception):
    """
    Raised when a sensor tries to preprocess for an unsupported model.
    """
    pass

class Sensor(ABC):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, "_" + k, v)

    """ Properties """
    @property
    def marker_shape(self) -> Optional[Tuple[int, int]]:
        return None

    @property 
    def supported_models(self) -> List[ModelType]:
        return []

    """ Required methods """
    @abstractmethod 
    def get_frame(self) -> Optional[Frame]:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def preprocess_for(self, model: ModelType, frame: Frame) -> Frame:
        pass

    """ Optional methods """
    def get_markers(self) -> Optional[Markers]:
        raise NotImplementedError()
