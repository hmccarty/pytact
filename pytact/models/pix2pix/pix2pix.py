import torch
import torch.nn as nn
from torch.optim import Adam
from torch.util.data import DataLoader
from pytact.models import Model
from pytact.types import ModelType
from typing import Tuple
from .networks import UnetGenerator, PatchGANDescrimator

class Pix2PixModel(Model):
    """
    Image translation network from raw to depth image.
    """
    
    model_type = ModelType.P2PGrad
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)

    def __init__(self, in_channels: int, out_channels: int, is_train: bool = True):
        super().__init__()
        self.device = torch.device(self.device_type) 
        self.G = UnetGenerator(in_channels, out_channels)
        
        self.is_train = is_train
        if self.is_train:
            self.D = PatchGANDescrimator(in_channels)

            # Optimizers
            self.G_opt = Adam(self.G.parameters(), lr=self.lr, betas=self.betas)
            self.D_opt = Adam(self.D.parameters(), lr=self.lr, betas=self.betas)

            # Loss
            self.l1_loss = nn.L1Loss()
            self.gan_loss = nn.BCEWithLogitsLoss()

    def __call__(self, x):
        return self.G(x.to(self.device))

    def run_epoch(self, dataloader: DataLoader):
        size = len(dataloader.dataset)
        
        # TODO: What do? 
        # model.train()
        
        for batch, (X, y) in enumerate(dataloader):
            X_real, y_real = X.to(self.device), y.to(self.device)

            # Create generated image
            y_fake = self.G(X_real)

            # Take backwards step on descriminator
            self.D.requires_grad = True
            self.D_opt.zero_grad()

            fake_score = self.D(torch.cat((X_real, y_fake), 1)) 
            fake_loss = self.gan_loss(fake_score, torch.tensor(-1.0).expand_as(fake_score))
            
            real_score = self.D(torch.cat((X_real, y_real), 1))
            real_loss = self.gan_loss(real_score, torch.tensor(1.0).expand_as(real_score))

            D_loss = (fake_loss + real_loss) * 0.5
            D_loss.backward()
            self.D_opt.step()

            # Take backwards step on generator
            self.D.requires_grad = False
            self.G_opt.zero_grad()

            G_loss = self.l1_loss(y_fake, y_real)
            G_loss += self.gan_loss(fake_score, torch.tensor(1.0).expand_as(fake_score))
            G_loss.backward()
            self.G_opt.step()
            
            if batch % 100 == 0:
                loss_log = f"G loss: {G_loss.item()}, D loss: {D_loss.item()} "
                loss_log += f"[{batch * len(X):>5d} / {size:>5d}]"
                print(loss_log)