from marker_util import Matching
from typing import Tuple
import numpy as np

from pytact.sensors import Sensor
from pytact.types import Flow, Markers

from pytact.tasks import Task

class FlowFromMarkers(Task):
    """
    Uses markers to compute a vector field. Can be used for estimating shear forces.

    Parameters
    ----------
    marker_shape: Tuple[int, int] 
        Number of markers per row and column
    marker_origin: Tuple[int, int]
        Location in frame of first marker
    marker_dist: Tuple[int, int] 
        Average row and column difference of the markers    
    """

    _error_threshold: float = 5.0

    def __init__(self, marker_shape: Tuple[int, int], marker_origin: Tuple[int, int], marker_dist: Tuple[int, int]):
        super().__init__()

        self.marker_shape = marker_shape
        self.marker_origin = marker_origin
        self.marker_dist = marker_dist
        self._match = Matching(
            self.marker_shape[0], self.marker_shape[1],
            30, marker_origin[0], marker_origin[1],
            marker_dist[0], marker_dist[1]
        )

    def reset_matching(self):
        del self._match
        self._match = Matching(
            self.marker_shape[0], self.marker_shape[1],
            30, self.marker_origin[0], self.marker_origin[1],
            self.marker_dist[0], self.marker_dist[1]
        )

    def __call__(self, sensor: Sensor) -> Flow:
        marker_shape = sensor.marker_shape
        if marker_shape is None:
            raise RuntimeError(f"Marker shape could not be retrieved from sensor: {sensor}")

        markers = sensor.get_markers()
        if markers is None:
            raise RuntimeError(f"Markers could not be retrieved from sensor: {sensor}")

        self._match.init(markers.markers)
        self._match.run()
        Ox, Oy, Cx, Cy, _ = self._match.get_flow()

        # Transform into shape: (n_markers, 2)
        Ox_t = np.reshape(np.array(Ox).flatten(), (len(Ox) * len(Ox[0]), 1))
        Oy_t = np.reshape(np.array(Oy).flatten(), (len(Oy) * len(Oy[0]), 1))
        ref_markers = Markers(marker_shape[0], marker_shape[1], np.hstack((Ox_t, Oy_t)))
        Cx_t = np.reshape(np.array(Cx).flatten(), (len(Cx) * len(Cx[0]), 1))
        Cy_t = np.reshape(np.array(Cy).flatten(), (len(Cy) * len(Cy[0]), 1))
        cur_markers = Markers(marker_shape[0], marker_shape[1], np.hstack((Cx_t, Cy_t)))

        self._flow = Flow(ref_markers, cur_markers)
        displacement: float = np.mean(self._flow.cur.markers - self._flow.ref.markers)
        if np.mean(displacement, dtype=np.float64) <= self._error_threshold: # hack to determine if calibration is incorrect
            self.reset_matching()
            return self.__call__(sensor)
        return self._flow