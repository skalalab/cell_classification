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
from torcheval.metrics.functional import binary_auprc, binary_accuracy

class CellDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list.copy()
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        donor = self.img_list[idx][:2]
        img_path = f"Images/{donor}_Cells/" + self.img_list[idx]
        
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
    def __init__(self, patience, min_delta=0.0):
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
            

def validate(device, model, early_stopper, early_loader, weight):
    model.eval()
    validation_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(early_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, weight=weight, reduction="sum").item()
            validation_loss += loss

    
    validation_loss /= len(early_loader.dataset)
    
    return early_stopper.validate(validation_loss)


def test(device, model, test_loader, weight):
    model.eval()
    
    total_pred = torch.tensor([]).type(torch.int64).to(device)
    total_truth = torch.tensor([]).type(torch.int64).to(device)
    count = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:   
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            pred = torch.swapaxes(output.data.max(1, keepdim=True)[1], 0, 1)[0].to(device)
            truth = target.data.view_as(pred).to(device)
            total_pred = torch.cat((total_pred, pred))
            total_truth = torch.cat((total_truth, truth))
            count += pred.eq(truth).sum().item()
            loss += F.nll_loss(output, target, weight=weight, reduction="sum").item()
            
    test_loss = loss / len(test_loader.dataset)
    accuracy = binary_accuracy(total_pred, total_truth).item()
    test_accuracy = count / len(test_loader.dataset)
    assert round(accuracy, 3) == round(test_accuracy, 3), f"acc: {accuracy}, test: {test_accuracy}"

    average_precision = binary_auprc(total_pred, total_truth).item()
            
    
    return {"loss": test_loss, "acc": accuracy, "ap": average_precision}
    


# set up cuda
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  

# load all the cell names
all_cells = {}

for i in range(1, 7):
    with open(f"Images/D{i}_Cells/originals.txt") as file:
        cells = []
        for line in file:
            if line == "":
                continue
            
            cells.append(line[:-1])
            
    all_cells[f"{i}"] = cells


# start the training
n_epochs = 1
all_donors = [1,2,3,5,6]
# batch_sizes = [4,8,16,32]
# learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [4,8]
learning_rates = [0.001, 0.01]

# get the best hyperparameters for all test donors
model_results = {"1": [], "2": [], "3": [], "5": [], "6": []}
for test_donor in all_donors:
    train_donors = all_donors.copy()
    train_donors.remove(test_donor)

    # loop through all hyperparameter combinations (grid-search)
    for bs in batch_sizes:
        for lr in learning_rates:
            # four fold cross validation by leaving test donor out    
            mean_acc = 0 
            mean_ap = 0
            mean_loss = 0
            for validation_donor in train_donors:
                # get train/validation/early-stop sets    
                train_cells = []
                for donor in train_donors:
                    if donor == validation_donor:
                        continue
                    
                    for cell in all_cells[f"{donor}"]:
                        train_cells.append(cell)  
            
                random.shuffle(train_cells)
                
                early_set = train_cells[:math.floor(len(train_cells) / 4)]
            
                train_set = []
                for i in range(math.floor(len(train_cells) / 4), len(train_cells)):
                    original_cell = train_cells[i]
                    train_set.append(original_cell)
                    train_set.append(original_cell.replace("no", "fh"))
                    train_set.append(original_cell.replace("no", "fv"))
                    train_set.append(original_cell.replace("no", "r90"))
                    train_set.append(original_cell.replace("no", "r180"))
                    train_set.append(original_cell.replace("no", "r270"))
                    
                validation_set = all_cells[f"{validation_donor}"]
        
                # load data into dataloader
                train_loader = torch.utils.data.DataLoader(CellDataset(train_set, ToTensor()),
                                                           batch_size=bs, shuffle=True)
                validation_loader = torch.utils.data.DataLoader(CellDataset(validation_set, ToTensor()),
                                                          batch_size=100, shuffle=True)
                early_loader = torch.utils.data.DataLoader(CellDataset(early_set, ToTensor()),
                                                           batch_size=100, shuffle=True)
            
                # print(f"test: {test_donor} | valid: {validation_donor}")
                # print(len(train_loader.dataset))
                # print(len(validation_loader.dataset))
                # print(len(early_loader.dataset))
            
                # set class weights
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
                optimizer = optim.Adam(model.parameters(), lr=lr)
                early_stopper = EarlyStopper(20)
                
                for i in range(n_epochs):
                    train(device, model, optimizer, train_loader, class_weight)    
                    stop_early = validate(device, model, early_stopper, early_loader, class_weight)    
                
                    if stop_early:
                        break;
                        
                results = test(device, model, validation_loader, class_weight)
                mean_loss += results["loss"]
                mean_acc += results["acc"]
                mean_ap += results["ap"]
                print(str(test_donor) + " | " + str(validation_donor) + ": " + str(results))
    
            # get the average of the four models in cross validation training 
            mean_loss /= len(train_donors)
            mean_acc /= len(train_donors)
            mean_ap /= len(train_donors)
            
            print(f"donor: {test_donor} | bs: {bs} | lr: {lr} | loss: {mean_loss} | acc: {mean_acc}| ap: {mean_ap} ")
            model_results[f"{test_donor}"].append({"bs": bs, "lr": lr, "loss": mean_loss, "acc": mean_acc, "ap": mean_ap})
     
        
# get the best model for all test donors
best_models = {}
print("\n\nbest models:")
for donor in all_donors:
    best_ap = 0
    best_bs = None
    best_lr = None
    for result in model_results[f"{donor}"]:
        print(f"donor {donor}: " + str(result))
        
        if result["ap"] > best_ap:
            best_ap = result["ap"]
            best_bs = result["bs"]
            best_lr = result["lr"]
            
    best_models[f"{donor}"] = {"bs": best_bs, "lr": best_lr}
    print(f"donor {donor}: best batch size: {best_bs} | best learning rate: {best_lr}")
            
    
# test the best model for each test donor
print("\n\ntest models:")
for test_donor in all_donors:
    # get optimal hyperparameters for this model
    hyperparameters = best_models[f"{donor}"]
    bs = hyperparameters["bs"]
    lr = hyperparameters["lr"]
    
    # get train/test/early-stop sets    
    train_cells = []
    for donor in all_donors:
        if donor == test_donor:
            continue
        
        for cell in all_cells[f"{donor}"]:
            train_cells.append(cell)  

    random.shuffle(train_cells)
    
    early_set = train_cells[:math.floor(len(train_cells) / 4)]

    train_set = []
    for i in range(math.floor(len(train_cells) / 4), len(train_cells)):
        original_cell = train_cells[i]
        train_set.append(original_cell)
        train_set.append(original_cell.replace("no", "fh"))
        train_set.append(original_cell.replace("no", "fv"))
        train_set.append(original_cell.replace("no", "r90"))
        train_set.append(original_cell.replace("no", "r180"))
        train_set.append(original_cell.replace("no", "r270"))
        
    test_set = all_cells[f"{test_donor}"]

    # load data into dataloader
    train_loader = torch.utils.data.DataLoader(CellDataset(train_set, ToTensor()),
                                               batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CellDataset(test_set, ToTensor()),
                                              batch_size=100, shuffle=True)
    early_loader = torch.utils.data.DataLoader(CellDataset(early_set, ToTensor()),
                                               batch_size=100, shuffle=True)

    # set class weights
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopper(20)
    
    for i in range(n_epochs):
        train(device, model, optimizer, train_loader, class_weight)    
        stop_early = validate(device, model, early_stopper, early_loader, class_weight)    
    
        if stop_early:
            break;
            
    results = test(device, model, test_loader, class_weight)
    print(f"donor {test_donor}: " + str(results))