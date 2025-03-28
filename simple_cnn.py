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
from collections import Counter

from torch.utils.data import Dataset
import pandas as pd
import tifffile as tiff
import numpy as np
from torchvision.transforms import ToTensor
import torch.multiprocessing as mp

class CellDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_dir + "/" + self.img_labels.iloc[idx, 0]
        
        with tiff.TiffFile(img_path) as tif:
            image = self.transform(tif.asarray().astype(np.uint8)).unsqueeze(0)
            
        label = 1 if self.img_labels.iloc[idx, 1].strip() == "True" else 0
        return image, label
    
n_epochs = 5
batch_size_train = 4
batch_size_test = 20    
learning_rate = 0.001
log_interval = 1

random_seed = 5
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_annotations = "Images/Train/labels.csv"
train_dir = "Images/Train"
test_annotations = "Images/Test/labels.csv"
test_dir = "Images/Test"

train_loader = torch.utils.data.DataLoader(CellDataset(train_annotations, train_dir, ToTensor()),
                                           batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(CellDataset(test_annotations, test_dir, ToTensor()),
                                          batch_size=batch_size_test, shuffle=False)

#%%       
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# fig = plt.figure()

# for i in range(6):
#     plt.subplot(2,3, i+1)
#     plt.tight_layout()
    
#     example_img = example_data[i][0]
#     example_img = torch.swapaxes(example_img, 0, 2)
#     example_img = torch.swapaxes(example_img, 0, 1)
    
#     plt.imshow(torch.sum(example_img, 2))
#     plt.title("Ground Truth: {}".format(example_targets[i]))

# plt.show(fig)

training_csv = pd.read_csv(train_annotations)
count = Counter(0 if 'quiescent' in training_csv.iloc[row, 0] else 1
                     for row in range(1, training_csv.shape[0]))

if count[1] > count[0]:
    class_weight = torch.tensor([count[1]/count[0], 1])
else:
    class_weight = torch.tensor([1, count[0]/count[1]])

#%%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=(17,3,3))
        self.conv2 = nn.Conv3d(10, 20, kernel_size=(5,3,3))
        self.fc1 = nn.Linear(63000, 120)
        self.fc2 = nn.Linear(120, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool3d(x, (4,2,2)))
        x = self.conv2(x)
        x = F.relu(F.max_pool3d(x, (4,2,2)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, -1)
    

    
simple_cnn = SimpleCNN()
optimizer = optim.Adam(simple_cnn.parameters(), lr=learning_rate)

# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = [[i*len(train_loader.dataset) for i in range(n_epochs + 1)]]

def train(epochs):
    simple_cnn.train()
    
    for i in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = simple_cnn(data)
            loss = F.nll_loss(output, target, weight=class_weight)
            loss.backward()
            optimizer.step()
        
            if batch_idx % log_interval == 0:
                print("{}: epoch {}: {}/{}: {}".format(mp.current_process().name, 
                    i+1, (batch_idx+1) * len(data), len(train_loader.dataset),
                    loss.item()))
                # train_losses.append(loss.item())
                # train_counter.append(((batch_idx+1)*batch_size_train))

#%%
def test():
    simple_cnn.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:   
            output = simple_cnn(data)
            test_loss += F.nll_loss(output, target, weight=class_weight, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
       
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
#%%

# if __name__ == "__main__":
#     # set up multi processing
#     num_processes = 2
#     simple_cnn.share_memory()
    
#     # start each process
#     test()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=train, args=(2,), name=f'Process-{rank}')
#         p.start()
#         processes.append(p)
#         print(f'Started {p.name}')
        
#     # wait for all processes to finish
#     for p in processes:
#         p.join()
#         print(f'Finished {p.name}')
#     test()


test()
train(n_epochs)
test()
    

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig
        
        