# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:49:42 2025

@author: chris
"""

import numpy as np
import tifffile as tiff
import sdt_reader
import os
from pathlib import Path
import visualizer
import math
import random
from glob import glob
import scipy
import matplotlib.pyplot as plt
import cv2

# information for cell
class Cell():
    def __init__(self, value, image, activation, name):
        self.value = value
        self.image = image
        self.activation = activation
        self.name = name
        self.entropy = 0


# split sdt into single cropped cells
def split_sdt(sdt, mask): 
    # get image data
    sdt_data = sdt_reader.read_sdt150(sdt)

    # get nonempty channel if needed
    if sdt_data.ndim == 4:
        for i in range(sdt_data.shape[0]):
            if np.count_nonzero(sdt_data[i]) == 0:
                continue
            
            sdt_data = sdt_data[i]
            break
     
    # get mask
    with tiff.TiffFile(mask) as tif:
        mask_data = tif.asarray()
        
    # get cell values
    cell_values = np.unique(mask_data)
    cell_values = cell_values[1:]
    
    # single out each cell and find bounds
    cell_images = []
    for cell_value in cell_values:
        # mask each cell
        cell_image = np.copy(sdt_data)
        
        upper = None
        left = cell_image.shape[0]
        right = 0
        for row in range(mask_data.shape[0]):
            for col in range(mask_data.shape[1]):
                if mask_data[row][col] != cell_value:
                    cell_image[row][col][:] = 0
                else:
                    # set upper and lower
                    if upper is None:
                        upper = row
                    
                    lower = row
                    
                    # set left and right
                    if col < left:
                        left = col
                        
                    if col > right:
                        right = col
                
        cropped_image = cell_image[upper:lower+1, left:right+1, :]
        
        # create Cell for this cell
        name = Path(sdt).name[:-4]
        activation = "act" in name
        cell_images.append(Cell(cell_value, cropped_image, activation, name))
    
    return cell_images    

# get max size
def max_size(cells):
    # get largest dimension as size
    cells.sort(key=lambda x: max(x.image.shape[0], x.image.shape[1]), reverse=True)
    size = max(cells[0].image.shape[0], cells[0].image.shape[1])
    
    # make size even
    if size % 2 != 0:
        size += 1
        
    return size


# pad cells
def pad_cells(cells, size):
    # pad all images to be size x size
    for cell in cells:
        vert_padding = size - cell.image.shape[0]
        side_padding = size - cell.image.shape[1]
        cell.image = np.pad(cell.image, 
                            ((math.floor(vert_padding/2),math.ceil(vert_padding/2)), 
                             (math.floor(side_padding/2),math.ceil(side_padding/2)), (0, 0)), 
                            'constant')   
    

# get entropy of cell
def calculate_entropy(cell):
    # get the time frame with highest intensity
    timeframes = np.sum(np.sum(cell.image, 0), 0)
    
    max_intensity = 0
    max_frame = 0
    for i in range(timeframes.shape[0]):
        intensity = timeframes[i]
        
        if intensity > max_intensity:
            max_intensity = intensity
            max_frame = i

    # calculate entropy for max time frame
    cell_max = np.max(cell.image[:,:,max_frame])
    cell_hist, bins = np.histogram(cell.image[:,:,max_frame], bins=cell_max + 1, range=(0,cell_max+1))
    cell.entropy = scipy.stats.entropy(cell_hist)
        

# filter cells by entropy
def filter_cells(cells, q):
    # get entropies
    entropies = []
    for cell in cells:
        calculate_entropy(cell)
        entropies.append(cell.entropy)

    cells.sort(key=lambda x: x.entropy)

    # remove the lowest q*100% cells
    mean = np.mean(entropies)
    std = np.std(entropies)
    cutoff = scipy.stats.norm.ppf(q, loc=mean, scale=std)
    print(mean)
    print(std)
    print(cutoff)

    for i in range(len(cells)):
        if cells[i].entropy >= cutoff:
            filtered = cells[:i]
            cells = cells[i:]
            break
        
    if filtered == None:
        filtered = []
        
    return cells, filtered

        
# save cells to folder
def save_cells(cells, output): 
    
    # create output folder
    if not os.path.exists(output):
        os.makedirs(output)
        
    # make label csv
    with open(output + "/labels.csv", "w") as labels:
        labels.write("cell_image, cell_label\n")
        for cell in cells:
            tiff.imwrite(output + "/{}_cell{}.tif".format(cell.name, str(cell.value)), cell.image)
            labels.write("{}_cell{}.tif, {}\n".format(cell.name, str(cell.value), cell.activation))


# crop all cells
active = []

for file in glob("Images/*active.sdt"):
    cells = split_sdt(file, file.replace(".sdt",".tif"))
    
    for cell in cells:
        active.append(cell)
    
quiescent = []
    
for file in glob("Images/*quiescent.sdt"):
    cells = split_sdt(file, file.replace(".sdt",".tif"))
    
    for cell in cells:
        quiescent.append(cell)


# pad all cells
size = max(max_size(active), max_size(quiescent))
pad_cells(active, size)
pad_cells(quiescent, size)


# filter cells
active, filtered = filter_cells(active, 0.1)

for cell in filtered:
    visualizer.visualize_array(cell.image, "filtered")
    
quiescent, filtered = filter_cells(quiescent, 0.1)

for cell in filtered:
    visualizer.visualize_array(cell.image, "filtered")
    

# split into test/train
random.seed(10)
random.shuffle(active)
random.shuffle(quiescent)

active_split_ind = math.floor((len(active) / 5))
quiescent_split_ind = math.floor((len(quiescent) / 5))

test = active[:active_split_ind] + quiescent[:quiescent_split_ind]
train = active[active_split_ind:] + quiescent[quiescent_split_ind:]

    
# save to folders
# save_cells(test, "Images/Test")
# save_cells(train, "Images/Train")