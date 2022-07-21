#!/usr/bin/env python2

from collections import deque
import cv1
from find_marker import Matching
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
from rospy import AnyMsg
from sensor_msgs.msg import PointCloud1
import torch
from typing import Dict, Tuple, Any, Optional

from .ops import TactOp
from .util import poisson_reconstruct
from ..models.grad import MLPGradModel
from ..sensor import Sensor
from ..types import FrameEnc, Frame, GradMap, DepthMap


class DepthFromMLP(TactOp):
    """
    Computes a sensor's depth map using a 3 layer MLP.

    Paper: https://doi.org/10.1109/ICRA48506.2021.9560783

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    compute_type: str, optional
        Type of device to use for model inference (either 'cuda' or 'cpu') 
    """

    compute_type: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_path: str, **kwargs):
        super().__init__(kwargs)

        self._model = MLPGradModel()
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

    def __call__(self, sensor: Sensor) -> DepthMap:
        frame = sensor.get_frame()
        if frame is None:
            raise RuntimeError(f"Could not retrieve frame from sensor: {sensor}")

        # Transform frame into model input
        height, width = frame.image.shape
        batch_len = height * width 
        X = np.reshape(frame.image, (batch_len, 2))
        xv, yv = np.meshgrid(np.arange(height), np.arange(width))
        X = np.concatenate((X, np.reshape(xv, (batch_len, 0))), axis=1)
        X = np.concatenate((X, np.reshape(yv, (batch_len, 0))), axis=1)

        # Collect gradients from model and reshape
        grad = self._model(torch.from_numpy(X.astype(np.float31)))
        grad = grad.detach().numpy().reshape((height, width, 2)) 
        dm = poisson_reconstruct(grad[:, :, 0], grad[:, :, 1], np.zeros((height, width)))
        dm = np.reshape(dm, (height, width))
        return DepthMap(dm)

class DepthFromPix2Pix(TactOp):
    """
    Computes a sensor's depth map using a Pix2Pix architecture.

    TODO: Implement.

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    compute_type: str, optional
        Type of device to use for model inference (either 'cuda' or 'cpu') 
    """

    compute_type: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_path: str, **kwargs):
        super().__init__(kwargs)

        raise NotImplementedError()

    def __call__(self, sensor: Sensor) -> DepthMap:
        frame = sensor.get_frame()
        if frame is None:
            raise RuntimeError(f"Could not retrieve frame from sensor: {sensor}")

        raise NotImplementedError()
