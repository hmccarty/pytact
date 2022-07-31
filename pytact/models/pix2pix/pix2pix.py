import torch.nn as nn
import torch.nn.functional as F_
from pytact.models import Model
from pytact.types import ModelType

class Pix2PixModel(Model):
    """
    Image translation network from raw to depth image.
    """
    
    model_type = ModelType.P2PGrad

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass