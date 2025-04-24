# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:48:39 2025

@author: chris
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

import math
from torch.utils.data import Dataset
import pandas as pd
import tifffile as tiff
import numpy as np
from torchvision.transforms import ToTensor
from torcheval.metrics.functional import binary_auprc

class CellDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        donor = self.img_list[idx][:2]
        img_path = "Images/{}_Cells/".format(donor) + self.img_list[idx]
        
        with tiff.TiffFile(img_path) as tif:
            image = self.transform(tif.asarray().astype(np.uint8)).unsqueeze(0)
            
        label = 1 if "act" in self.img_list[idx] else 0
        return image, label


    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(16,7,7), stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(4,2,2))
        self.fc1 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(F.max_pool3d(x, (4,2,2)))
        x = self.conv2(x)
        x = F.leaky_relu(F.max_pool3d(x, (4,2,2)))
        
        # global averaging pool
        x = F.avg_pool3d(x, kernel_size=(6,7,7))
        
        x = x.flatten(1)
        x = F.leaky_relu(self.fc1(x))
        return F.log_softmax(x, -1)
    

class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def validate(self, validation_loss):
        if self.best_loss - validation_loss > self.min_delta:
            self.counter = 0
            self.best_loss = validation_loss

            return False
        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True



def train(device, model, optimizer, train_loader, weight):
    # train one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # transfer data tensors to GPU
        data = data.to(device)
        target = target.to(device)
        
        # gradient descent and backpropagation
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, weight=weight)
        loss.backward()
        optimizer.step()
            

def validate(device, model, early_stopper, validation_loader, weight):
    model.eval()
    validation_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, weight=weight, reduction="sum").item()
    
    validation_loss /= len(validation_loader.dataset)
    
    return early_stopper.validate(validation_loss), validation_loss


def test(device, model, test_loader, weight):
    model.eval()
    test_loss = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    
    with torch.no_grad():
        for data, target in test_loader:   
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            # get loss
            test_loss = F.nll_loss(output, target, weight=weight, reduction="mean").item()
            
            # get accuracy
            pred = output.data.max(1, keepdim=True)[1]
            truth = target.data.view_as(pred)
            accuracy = pred.eq(truth).sum() / len(test_loader.dataset)

            # get average precision
            average_precision = binary_auprc(pred, truth).item()
            
    
    return {"acc": accuracy, "ap": average_precision}
    


# set up cuda
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  

# load all the cell names
all_cells = []

for i in range(1, 7):
    with open("Images/D{}_Cells/originals.txt".format(i)) as file:
        cells = []
        for line in file:
            cells.append(line[:-1])
            
    all_cells.append(cells)


# start the training
n_epochs = 100
all_donors = [1,2,3,5,6]
batch_sizes = [4,8,16,32]
learning_rates = [0.0001, 0.001, 0.01, 0.1]
results = []

# outer loops is hyperparameter grid search
for learning_rate in learning_rates:
    for batch_size_train in batch_sizes:
        
        # inner loops is four-fold cross validation
        for test_donor in all_donors:
            donors = all_donors.copy()
            donors.remove(test_donor)
            mean_ap = 0
            mean_acc = 0
            
            for validation_donor in donors:
                train_donors = donors.copy()
                train_donors.remove(validation_donor)
                
                 # get train/test/validation sets
                test_set = all_cells[validation_donor]
            
                train_cells = []
                for donor in train_donors:
                    train_cells.append(all_cells[donor])    
            
                random.shuffle(train_cells)
                
                validation_set = train_cells[:math.floor(len(train_cells) / 4)]
            
                train_set = []
                for i in range(math.floor(len(train_cells) / 4), len(train_cells)):
                    original_cell = train_cells[i]
                    train_cells.append(original_cell)
                    train_cells.append(original_cell.replace("no", "fh"))
                    train_cells.append(original_cell.replace("no", "fv"))
                    train_cells.append(original_cell.replace("no", "r90"))
                    train_cells.append(original_cell.replace("no", "r180"))
                    train_cells.append(original_cell.replace("no", "r270"))
                
                # load data into dataloader
                train_loader = torch.utils.data.DataLoader(CellDataset(train_set, ToTensor()),
                                                           batch_size=batch_size_train, shuffle=True)
                test_loader = torch.utils.data.DataLoader(CellDataset(test_set, ToTensor()),
                                                          batch_size=len(test_set), shuffle=True)
                validation_loader = torch.utils.data.DataLoader(CellDataset(validation_set, ToTensor()),
                                                           batch_size=len(validation_set), shuffle=True)
            
                # class weights
                active_count = 0
                quiescent_count = 0
                for cell in train_set:
                    if "act" in cell:
                        active_count += 1
                    else:
                        quiescent_count += 1
                    
                if active_count > quiescent_count:
                    class_weight = torch.tensor([active_count/quiescent_count, 1]).to(device)
                else:
                    class_weight = torch.tensor([1, quiescent_count/active_count]).to(device)
                
                # train the model
                model = SimpleCNN().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                early_stopper = EarlyStopper(10, 0.0)
                
                for i in range(n_epochs):
                    train(device, model, optimizer, train_loader, class_weight)    
                    stop_early, validation_loss = validate(device, model, early_stopper, validation_loader, class_weight)    
                
                    if stop_early:
                        break;
                        
                result = test(device, model, test_loader, class_weight)
                mean_ap += result["ap"]
                mean_acc += result["acc"]
                
            # mean average precision    
            mean_ap /= 4
            mean_acc /= 4