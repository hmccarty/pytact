#!/usr/bin/env python3

import cv2
from marker_util import Matching
import numpy as np
from typing import Dict, Tuple, Any, Optional

from pytact.sensor import Sensor
from pytact.types import FrameEnc, Flow

from .ops import TactOp
from .util import find_markers

class FlowFromMarkers(TactOp):
    """
    Uses markers to compute a vector field. Can be used for estimating shear forces.

    Parameters
    ----------
    marker_origin: Tuple[int, int]
        Location in frame of first marker
    marker_dist: Tuple[int, int] 
        Average row and column difference of the markers    
    threshold_block_size: int, optional
    threshold_neg_bias: int, optional
    marker_neighborhood_size: int, optional
    """

    threshold_block_size: int = 17
    threshold_neg_bias: int = 25
    marker_neighborhood_size: int = 20

    def __init__(self,  marker_origin: Tuple[int, int], marker_dist: Tuple[int, int], **kwargs):
        super().__init__(kwargs)


        self._match = Matching(
            self.marker_shape[0], self.marker_shape[1],
            self.matching_fps, cfg["x0"], cfg["y0"], cfg["dx"], cfg["dy"]
        )

    def __call__(self, sensor: Sensor) -> Flow:
        if sensor.marker_shape is None:
            raise RuntimeError(f"FlowFromMarkers: No markers found for sensor: {sensor}")

        frame = sensor.get_frame()
        if frame is None:
            raise RuntimeError(f"FlowFromMarkers: Could not retrieve frame from sensor: {sensor}")

        cur = find_markers(frame, self.threshold_block_size, self.threshold_neg_bias, self.marker_neighborhood_size)
        # Convert to gelsight dataclass
        self._markers = GelsightMarkers(im.shape[1], im.shape[0], xy)


# class FlowProc(GelsightProc):
#     """
#     Approximates displacement of markers. 

#     execute() -> GelsightFlowStamped msg

#     Params:
#       - x0 (Required)
#       - y0 (Required)
#       - dx (Required)
#       - dy (Required)
#       - marker_shape (default: (10, 12))
#       - matching_fps (default: 25)
#       - flow_scale (default: 5)
#     """

#     # Parameter defaults
#     marker_shape: Tuple[int, int] = (14, 10) 
#     matching_fps: int = 25
#     flow_scale: float = 5
#     error_threshold: float = -50

#     def __init__(self, markers: MarkersProc, cfg: Dict[str, Any]):
#         super().__init__()
#         self._markers: MarkersProc = markers
#         self._flow: Optional[GelsightFlow] = None
#         self._cfg: Dict[str, Any] = cfg

#         if not all(param in cfg for param in ["x0", "y0", "dx", "dy"]):
#             raise RuntimeError("FlowProc: Missing marker configuration.")

#         if "flow_scale" in cfg:
#             self.flow_scale = cfg["flow_scale"]

#         self._match = Matching(
#             self.marker_shape[0], self.marker_shape[1],
#             self.matching_fps, cfg["x0"], cfg["y0"], cfg["dx"], cfg["dy"]
#         )
    
#     def reset_matching(self):
#         del self._match
#         self._match = Matching(
#             self.marker_shape[0], self.marker_shape[1],
#             self.matching_fps, self._cfg["x0"], self._cfg["y0"], self._cfg["dx"], self._cfg["dy"]
#         )

#     def execute(self) -> GelsightFlowStampedMsg:
#         gsmarkers = self._markers.get_markers()  
#         if gsmarkers: 
#             self._match.init(gsmarkers.markers)
            
#             self._match.run()
#             Ox, Oy, Cx, Cy, _ = self._match.get_flow()

#             # Transform into shape: (n_markers, 2)
#             Ox_t = np.reshape(np.array(Ox).flatten(), (len(Ox) * len(Ox[0]), 1))
#             Oy_t = np.reshape(np.array(Oy).flatten(), (len(Oy) * len(Oy[0]), 1))
#             ref_markers = GelsightMarkers(self.marker_shape[0], self.marker_shape[1], np.hstack((Ox_t, Oy_t)))
#             Cx_t = np.reshape(np.array(Cx).flatten(), (len(Cx) * len(Cx[0]), 1))
#             Cy_t = np.reshape(np.array(Cy).flatten(), (len(Cy) * len(Cy[0]), 1))
#             cur_markers = GelsightMarkers(self.marker_shape[0], self.marker_shape[1], np.hstack((Cx_t, Cy_t)))

#             self._flow = GelsightFlow(ref_markers, cur_markers)
#             vec_field = self._flow.cur.markers - self._flow.ref.markers
#             if np.mean(vec_field) <= self.error_threshold: # hack to determine if calibration is incorrect
#                 raise ProcExecutionError("Marker flow is uncalibrated! Ensure all markers are detected in marker_image")
#                 self.reset_matching()

#     def get_flow(self) -> Optional[GelsightFlow]:
#         return self._flow

#     def get_ros_type(self) -> GelsightFlowStampedMsg:
#         return GelsightFlowStampedMsg

#     def get_ros_msg(self) -> GelsightFlowStampedMsg:
#         return self._flow.get_ros_msg_stamped()

# class DrawMarkersProc(GelsightProc):
#     """
#     Reads from stream and markers, then returns image with markers drawn.

#     execute() -> Image msg
#     """

#     encoding: str = "bgr8"    
#     marker_color: Tuple[int, int, int] = (255, 0, 0)
#     marker_radius: int = 2
#     marker_thickness: int = 2

#     def __init__(self, stream: ImageProc, markers: MarkersProc):
#         super().__init__()
#         self._stream: ImageProc = stream
#         self._markers: MarkersProc = markers
#         self._frame: Optional[np.ndarray] = None

#     def execute(self):
#         frame = self._stream.get_frame()
#         gsmarkers = self._markers.get_markers()
#         if gsmarkers is None:
#             return None

#         for i in range(gsmarkers.markers.shape[0]):
#             p0 = (int(gsmarkers.markers[i, 0]), int(gsmarkers.markers[i, 1]))
#             frame = cv2.circle(frame, p0, self.marker_radius,
#                 self.marker_color, self.marker_thickness)
        
#         self._frame = frame

#     def get_ros_type(self) -> Image:
#         return Image

#     def get_ros_msg(self) -> Image:
#         return CvBridge().cv2_to_imgmsg(self._frame, self.encoding)

# class DrawFlowProc(GelsightProc):
#     """
#     Reads from stream and flow, then returns image with flow.
    
#     execute() -> Image msg 
#     """

#     encoding: str = "bgr8"
#     arrow_color: Tuple[int, int, int] = (0, 255, 0)
#     arrow_thickness: int = 2
#     arrow_scale: int = 5

#     def __init__(self, stream: ImageProc, flow: FlowProc):
#         super().__init__()
#         self._stream: ImageProc = stream
#         self._flow: FlowProc = flow
#         self._frame: Optional[np.ndarray] = None

#     def execute(self):
#         frame = self._stream.get_frame()
#         flow = self._flow.get_flow()
#         if flow is None:
#             return

#         for i in range(flow.ref.markers.data.shape[0]):
#             p0 = (int(flow.ref.markers.data[i, 0]), int(flow.ref.markers.data[i, 1]))
#             p1 = (int(((flow.cur.markers.data[i, 0] - p0[0]) * self.arrow_scale) + p0[0]),
#                   int(((flow.cur.markers.data[i, 1] - p0[1]) * self.arrow_scale) + p0[1]))
#             frame = cv2.arrowedLine(frame, p0, p1,
#                 self.arrow_color, self.arrow_thickness)

#         self._frame = frame

#     def get_ros_type(self) -> Image:
#         return Image

#     def get_ros_msg(self) -> Image:
#         return CvBridge().cv2_to_imgmsg(self._frame, self.encoding)