import torch
import torch.nn as nn

class UnetGenerator(nn.Module):

    dropout_p = 0.5

    def __init__(self, input_channels: int, output_channels: int,  use_dropout=True):
        # Setup Unet downsampling 
        channels = [64, 128, 256] + [512]*5
        self.enc = [nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 4, stride=2),
            nn.LeakyReLU(negative_slope=0.2)
        )]
        for i in range(len(channels)-1):
            self.enc.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        # Setup Unet upsampling
        channels = [512]*2 + [1024]*4 + [512, 256, 128]
        self.denc = []
        for i in range(len(channels)-1):
            layer = [nn.ConvTranspose2d(channels[i], channels[i+1], 4, stride=2)]
            layer.append(nn.BatchNorm2d(channels[i+1]))
            if i < 3 and use_dropout:
                layer.append(nn.Dropout(self.dropout_p))
            layer.append(nn.ReLU())
            self.denc.append(nn.Sequential(*layer))
        self.denc.append(nn.Sequential(
            nn.ConvTranspose2d(channels[-1], output_channels, 4, stride=2),
            nn.Tanh()
        ))

    def forward(self, x):
        # Pass inputs through encoder and store activations
        act = [x]
        for layer in self.enc:
            act.append(layer(act[-1]))

        # Pass previous and skip activations through decoder
        act = act[1:]
        last_act = act[-1]
        for i in range(1, len(self.denc)-1):
            last_act = self.denc[i](torch.cat((act[len(act) - i], last_act), 1))

        # Return output under expected output channels
        return last_act

class PatchGANDescriminator(nn.Module):

    def __init__(self, input_channels: int):
        channels = [64, 128, 256, 512]
        self.net = [nn.Sequential(
            nn.Conv2d(input_channels, channels[0]),
            nn.LeakyReLU(negative_slope=0.2)
        )]
        for i in range(len(channels)-1):
            self.net.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(negative_slope=0.2)
            ))
        self.net.append(nn.Sequential(
            nn.Conv2d(channels[-1], 1, 4, stride=2),
            nn.Sigmoid()
        ))

    def forward(self, x):
        last_act = x
        for layer in self.net:
            last_act = layer(last_act)
        return last_act