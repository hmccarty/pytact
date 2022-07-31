import math
import numpy as np
import torch
import scipy

from pytact.models import Pixel2GradModel
from pytact.sensors import Sensor
from pytact.types import DepthMap, ModelType

from .tasks import Task

def poisson_reconstruct(gradx, grady, boundarysrc): 
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt

    return result

class DepthFromLookup(Task):
    """
    Computes a sensor's depth map using a 3-layer MLP which learned
    the lookup table for each pixel's gradient.

    Paper: https://doi.org/10.1109/ICRA48506.2021.9560783

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)

        self._model = Pixel2GradModel()
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

    def __call__(self, sensor: Sensor) -> DepthMap:
        frame = sensor.get_frame()
        if frame is None:
            raise RuntimeError(f"Could not retrieve frame from sensor: {sensor}")

        # Transform frame into model input
        frame = sensor.preprocess_for(ModelType.Pixel2Grad, frame)
        height, width = frame.image.shape
        batch_len = height * width 
        X = frame.image.reshape((batch_len, 2))
        xv, yv = np.meshgrid(np.arange(height), np.arange(width))
        X = np.concatenate((X, np.reshape(xv, (batch_len, 0))), axis=1)
        X = np.concatenate((X, np.reshape(yv, (batch_len, 0))), axis=1)

        # Collect gradients from model and reshape
        grad = self._model(torch.from_numpy(X.astype(np.float32)))
        grad = grad.detach().numpy().reshape((height, width, 2)) 
        dm = poisson_reconstruct(grad[:, :, 0], grad[:, :, 1], np.zeros((height, width)))
        dm = np.reshape(dm, (height, width))
        return DepthMap(dm)

class DepthFromPix2Pix(Task):
    """
    Computes a sensor's depth map using a Pix2Pix model.

    TODO: Implement.

    Parameters
    ----------
    model_path: str
        Path to model parameters; must match MLPGradModel in models/.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        raise NotImplementedError()

    def __call__(self, sensor: Sensor) -> DepthMap:
        frame = sensor.get_frame()
        if frame is None:
            raise RuntimeError(f"Could not retrieve frame from sensor: {sensor}")

        raise NotImplementedError()
