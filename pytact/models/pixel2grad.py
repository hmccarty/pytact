import torch.nn as nn
import torch.nn.functional as F_

from .model import Model
from pytact.types import ModelType

class Pixel2GradNetwork(nn.Module):
    dropout_p = 0.05

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = F_.relu(self.fc1(x))
        x = self.drop(x)
        x = F_.relu(self.fc2(x))
        x = self.drop(x)
        x = F_.relu(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)

class Pixel2GradModel(Model):
    """
    A 3-layer MLP to convert visuo-tactile pixels into depth gradients.

    Architecture: 5 (R, G, B, x, y) -> 64 -> 64 -> 64 -> 2 (gx, gy)
    """
    
    model_type = ModelType.Pixel2Grad

    def __init__(self):
        self.net = Pixel2GradNetwork()

    def forward(self, x):
        return self.net(x)