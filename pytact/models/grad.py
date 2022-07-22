import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F_

class MLPGradModel(nn.Module):
    """
    A 3-layer MLP to convert visuo-tactile pixels into depth gradients.

    Architecture: 5 (R, G, B, x, y) -> 64 -> 64 -> 64 -> 2 (gx, gy)
    """

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
        