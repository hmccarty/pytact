import torch
import torch.nn as nn
import torch.nn.functional as F_

class UnetGenerator(nn.Module):

    dropout_p = 0.5

    def __init__(self, input_channels: int, output_channels: int,  use_dropout=True):
        # Setup Unet downsampling 
        channels = [64, 128, 256] + [512]*5
        self.enc = [nn.Conv2d(input_channels, 64, 4, stride=2)]
        for i in range(len(channels)-1):
            self.enc.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2),
                nn.BatchNorm2d(channels[i+1])
            ))

        # Setup Unet upsampling
        channels = [512]*2 + [1024]*4 + [512, 256, 128, output_channels]
        self.denc = []
        for i in range(len(channels)-1):
            layer = [nn.ConvTranspose2d(channels[i], channels[i+1], 4, stride=2)]
            if i < 3:
                layer.append(nn.BatchNorm2d(channels[i+1]))
            if use_dropout:
                layer.append(nn.Dropout(self.dropout_p))
            self.denc.append(layer)

    def forward(self, x):
        # Pass inputs through encoder and store activations
        act = [x]
        for layer in self.enc:
            if isinstance(layer, nn.Conv2d):

            elif isinstance(layer, nn.BatchNorm2d):
                continue
            act.append(F_.leaky_relu(layer(act[-1]), negative_slope=0.2))

        # Pass skip and previous activations through decoder
        act = act[1:]
        last_act = F_.relu(self.denc[0](act[-1]))
        for i in range(1, len(self.denc)-1):
            last_act = F_.relu(self.denc[i](torch.cat((act[len(act) - i], last_act), 1)))

        # Return output under expected output channels
        return F_.tanh(self.denc[-1](torch.cat((act[0], last_act), 1)))

class PatchGANDescriminator(nn.Module):

    def __init__(self, input_channels: int):
        channels = [64, 128, 256, 512]
        self.net = [nn.Conv2d(input_channels, 64)]
        for i in range(len(channels)-1):
            self.net.append(
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2),
                nn.BatchNorm2d(channels[i+1])
            )

    def forward(self, x):
        y = self.net[0](x)
        for 