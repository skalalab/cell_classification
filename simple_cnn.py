# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:48:39 2025

@author: chris
"""

#%%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from torch.utils.data import Dataset
import pandas as pd
import tifffile as tiff

class CellDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.image_dir + "/" + self.img_labels.iloc[idx, 0]
        
        with tiff.TiffFile(img_path) as tif:
            image = torch.to_tensor(tif.asarray())
            
        label = self.img_labels.iloc[idx, 1]
        
        return image, label
    

batch_size_train = 10
batch_size_test = 48

train_annotations = "Images/Train/labels.csv"
train_dir = "Images/Train"
test_annotations = "Images/Test/labels.csv"
test_dir = "Images/Test"

train_loader = torch.utils.data.DataLoader(CellDataset(train_annotations, train_dir))
test_loader = torch.utils.data.DataLoader(CellDataset(test_annotations, test_dir))
        
            
        
        