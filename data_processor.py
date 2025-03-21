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
        cell.image = np.pad(cell.image, ((math.floor(vert_padding/2),math.ceil(vert_padding/2)), (math.floor(side_padding/2),math.ceil(side_padding/2)), (0, 0)), 'constant')   
    
# compute entropy
# def compute_entropy(img_name):
#     img = cv2.imread(img_name, 0)
    
#     cur_hist = cv2.calcHist([img],[0], None, [256], [0,256]).flatten()

#     # Remove zeroes
#     hist_no_zero = np.array([i for i in cur_hist if i != 0])

#     # Normalize the hist so it sums to 1
#     hist_no_zero = hist_no_zero / np.size(img)

#     # Compute the entropy
#     log_2 = np.log(hist_no_zero) / np.log(2)
#     entropy = -1 * np.dot(hist_no_zero, log_2)
#     return entropy


# filter cells based on entropy thresholding
def filter_cells(cells):
    # get entropy for each cell
    for cell in cells:
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
        cell_hist = np.copy(cell.image[:,:,max_frame])
        cell_max = np.max(cell_hist)
        cell_hist = cell_hist / cell_max
        cell_hist = cell_hist * 255
        # cell_hist, bins = np.histogram(cell_hist, bins=256, range=(0,256)) 
        # plt.bar(bins[:-1],cell_hist,width=1)
        plt.show()
        cell_hist =  cell_hist[cell_hist != 0]
        # plt.bar([i for i in range(cell_hist.shape[0])],cell_hist,width=1)
        # plt.show()
        cell_hist = cell_hist / np.size(cell.image[:,:,max_frame])
        cell.entropy = scipy.stats.entropy(cell_hist)
        
    cells.sort(key=lambda x: x.entropy)

    

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


# process
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

size = max(max_size(active), max_size(quiescent))
print(size)

pad_cells(active, size)
pad_cells(quiescent, size)


# split into test/train
random.seed(10)
random.shuffle(active)
random.shuffle(quiescent)

active_split_ind = math.floor((len(active) / 5))
quiescent_split_ind = math.floor((len(quiescent) / 5))

test = active[:active_split_ind] + quiescent[:quiescent_split_ind]
train = active[active_split_ind:] + quiescent[quiescent_split_ind:]

filter_cells(test)
    
# save to folders
# save_cells(test, "Images/Test")
# save_cells(train, "Images/Train")