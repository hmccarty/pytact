import torch.nn as nn
from torch.utils.data import DataLoader
from pytact.models import Model
from pytact.types import ModelType
from .networks import UnetGenerator, PatchGANDescriminator

class Pix2PixModel(Model):
    """
    Image translation network from raw to depth image.
    """
    
    model_type = ModelType.P2PGrad
    lr: float = 2e-4
    beta_1: float = 0.5
    beta_2: float = 0.999

    def __init__(self, in_channels: int, out_channels: int, train: bool = True):
        super().__init__()
        self.gen = UnetGenerator(in_channels, out_channels)
        
        self.train = train
        if train:
            self.desc = PatchGANDescriminator(in_channels)

    def __call__(self, x):
        pass

    def run_epoch(self, dataloader: DataLoader):
        pass