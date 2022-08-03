import torch.nn as nn
from torch.utils.data import DataLoader
from pytact.types import ModelType
from .model import Model

class Pixel2GradNetwork(nn.Module):
    dropout_p = 0.05

    def __init__(self):
        super().__init__()

        sizes = [5, 64, 64, 64]
        self.net = []
        for i in range(len(sizes)-1):
            self.net.append(nn.Sequential(
                nn.Linear(sizes[i], sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            ))
        self.net.append(nn.Sequential(
            nn.Linear(sizes[-1], 2),
            nn.ReLU()
        ))

    def forward(self, x):
        last_act = x
        for layer in self.net:
            last_act = layer(last_act)
        return last_act

class Pixel2GradModel(Model):
    """
    A 3-layer MLP to convert visuo-tactile pixels into depth gradients.

    Architecture: 5 (R, G, B, x, y) -> 64 -> 64 -> 64 -> 2 (gx, gy)
    """
    
    model_type = ModelType.Pixel2Grad

    def __init__(self):
        self.net = Pixel2GradNetwork()

    def __call__(self, x):
        return self.net(x)

    def run_epoch(self, dataloader: DataLoader):
        pass