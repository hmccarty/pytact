#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pytact
from torch.utils.data import Dataset
import torchvision.io as io
import pandas as pd
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Trains a pix2pix model for depth detection using dataset from the create_grad_dataset script.")
# parser.add_argument('input_path', type=str, help='Path to dataset')
parser.add_argument('--output_path', type=str, dest='output',
    default=os.getcwd(), help='Path to save model parameters')
parser.add_argument('--train_split', type=float, dest='train_split', default=0.8,
    help='Percentage of dataset to allocate as training data')
parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=10)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=64)
parser.add_argument('--learning_rate', type=int, dest='learning_rate', default=1e-3)
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], dest='device',
    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Global parameters
device = torch.device(args.device)

output_path = args.output
if not os.path.exists(output_path):
    print("Output folder doesn't exist, will create it.")
    os.makedirs(output_path)
output_file = output_path + f"/model-{dt.now().strftime('%H-%M-%S')}.pth"

# Create dataloader
input_path = "/home/yu/rgbt_dataset/kaist-cvpr15/images/set00/V000"

# Create dataset
class Pix2PixDataset(Dataset):
    def __init__(self, input_path, transform=None):
        self.rgb_path = input_path + "/visible"
        self.t_path = input_path + "/lwir"
        
        self.len = len([name for name in os.listdir(self.rgb_path) if os.path.isfile(self.rgb_path + "/" + name)])
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = io.read_image(f"{self.rgb_path}/I{idx:05}.jpg")
        y = io.read_image(f"{self.t_path}/I{idx:05}.jpg")
        return X, y    

dataset = Pix2PixDataset(input_path)

train_size = int(len(dataset) * args.train_split)
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=args.batch_size)

# Initiate model and optimizer
model = pytact.models.Pix2PixModel().to(device)

# Train model
epochs = args.num_epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.run_epoch(train_dataloader)
print("Done!")

model.save(output_file)