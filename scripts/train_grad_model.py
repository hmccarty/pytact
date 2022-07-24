#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pytact
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Trains a pixel->grad MLP model using dataset from the create_grad_dataset script.")
parser.add_argument('input_path', type=str, help='Path to dataset')
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

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

output_path = args.output
if not os.path.exists(output_path):
    print("Output folder doesn't exist, will create it.")
    os.makedirs(output_path)
output_file = output_path + f"/model-{dt.now().strftime('%H-%M-%S')}.pth"

# Create dataset
class Pixel2GradDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.labels.iloc[idx, 1:6].to_numpy()
        y = self.labels.iloc[idx, 6:].to_numpy()
        return X.astype(np.float32), y.astype(np.float32)    

dataset = Pixel2GradDataset(args.input_path)

train_size = int(len(dataset) * args.train_split)
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=args.batch_size)

# Initiate model and optimizer
model = pytact.models.MLPGradModel().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train model
epochs = args.num_epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), output_file)