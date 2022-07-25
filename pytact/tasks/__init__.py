from .tasks import Task
from .flow import FlowFromMarkers
from .depth import DepthFromLookup, DepthFromPix2Pix

__all__ = ['Task', 'FlowFromMarkers', 'DepthFromLookup', 'DepthFromPix2Pix']