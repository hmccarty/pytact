from abc import ABC, abstractmethod
from tkinter import W
from torch.utils.data import DataLoader

class Model(ABC):

    def __call__(self, x):
        raise NotImplementedError("Model: call dunder not implemented")

    @abstractmethod
    def run_epoch(self, dataloader: DataLoader):
        pass