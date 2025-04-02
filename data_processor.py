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
    def __init__(self, value, image, activation, filename, augmentation="no"):
        self.value = value
        self.image = image
        self.activation = activation
        self.filename = filename
        self.entropy = 0
        self.augmentation = augmentation

    def __repr__(self):
        return str(self.image.shape[0]) + " x " + str(self.image.shape[1])

# split sdt into single cropped cells
#
# param: sdt (string) - file path for sdt
# param: mask (string) - file path for mask
# return: list of Cells of cells in the sdt
def split_sdt(sdt, mask): 
    # get sdt data
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
    
    # mask each cell and find bounding box
    cell_images = []
    for cell_value in cell_values:
        cell_image = np.copy(sdt_data)
        
        # iterate through all pixels of image and zero out if not part of 
        # mask for cell
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
        
        # create Cell object for this cell
        filename = Path(sdt).name[:-4]
        activation = "act" in filename
        cell_images.append(Cell(cell_value, cropped_image, activation, filename))
    
    return cell_images    

# remove i largest cells and return the size of the new largest cell
#
# param: cells (list of Cells)
# param: i (int) - number of largest cells to remove
# returns: new list of Cells with largest i cells removed
# returns: largest size of cell in list
def remove_largest_cells(cells, i):
    # remove i largest cells
    cells.sort(key=lambda x: max(x.image.shape[0], x.image.shape[1]), reverse=True)
    cells = cells[i:]
    
    # get the size of largest cell
    size = max(cells[0].image.shape[0], cells[0].image.shape[1])
    
    # make size even
    if size % 2 != 0:
        size += 1
        
    return cells, size


# pad cells to be size x size
#
# param: cells (list of Cells)
# param: size (int) - size to be padded to
def pad_cells(cells, size):
    # pad all images to be size x size
    for cell in cells:
        vert_padding = size - cell.image.shape[0]
        side_padding = size - cell.image.shape[1]
        cell.image = np.pad(cell.image, 
                            ((math.floor(vert_padding/2),math.ceil(vert_padding/2)), 
                             (math.floor(side_padding/2),math.ceil(side_padding/2)), (0, 0)), 
                            'constant')   
    

# calculate entropy of a cell
#
# param: cell (Cell)
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
#
# param: cells (list of Cells)
# param: q (float) - lower tail probability
# return: list of cells that weren't filtered
# return: list of cells that were filtered
def filter_cells(cells, q):
    # get entropies
    entropies = []
    for cell in cells:
        calculate_entropy(cell)
        entropies.append(cell.entropy)

    # remove the lowest q*100%  of cells
    cells.sort(key=lambda x: x.entropy)
    mean = np.mean(entropies)
    std = np.std(entropies, ddof=1)
    cutoff = scipy.stats.norm.ppf(q, loc=mean, scale=std)


    for i in range(len(cells)):
        if cells[i].entropy >= cutoff:
            filtered = cells[:i]
            cells = cells[i:]
            break
        
    if filtered == None:
        filtered = []
        
    return cells, filtered

        
# add flipped horizontal/vertical, rotated 90, 180, 270 cells (counterclockwise)
#
# param: cells (list of Cells)
def augment_cells(cells):
    for i in range(len(cells)):
        # get Cell info
        value = cells[i].value
        image = cells[i].image
        activation = cells[i].activation
        filename = cells[i].filename
        
        # flip horizontal
        cells.append(Cell(value, np.fliplr(image), activation, filename, "fh"))
        
        # flip vertical
        cells.append(Cell(value, np.flipud(image), activation, filename, "fv"))
        
        # rotate 90
        cells.append(Cell(value, np.rot90(image), activation, filename, "r90"))
        
        # rotate 180
        cells.append(Cell(value, np.rot90(image, k=2), activation, filename, "r180"))

        # rotate 270
        cells.append(Cell(value, np.rot90(image, k=3), activation, filename, "r270"))


# save cells to folder and make csv of cells
#
# param: cells (list of Cells)
# param: output (string) - output path
def save_cells(cells, output): 
    # create output folder
    if not os.path.exists(output):
        os.makedirs(output)
        
    # make label csv
    with open(output + "/labels.csv", "w") as labels:
        labels.write("cell_image, cell_label\n")
        for cell in cells:
            tiff.imwrite(output + "/{}_cell{}_{}.tif".format(cell.filename, str(cell.value), cell.augmentation), cell.image)
            labels.write("{}_cell{}_{}.tif, {}\n".format(cell.filename, str(cell.value), cell.augmentation, cell.activation))




# now run the program

# crop all cells
print("start cropping...")
all_cells = []

for file in glob("Images/D1-3/*.sdt"):
    cells = split_sdt(file, file.replace(".sdt",".tif"))
    
    for cell in cells:
        all_cells.append(cell)

print("finish cropping...")
# remove i largest cells
print("start removing...")
all_cells, size = remove_largest_cells(all_cells, 0)
print("finish removing... size: " + str(size))

# pad cells
print("start padding...")
pad_cells(all_cells, size)
print("finish padding...")

# filter cells
print("start filtering...")
active = [cell for cell in all_cells if cell.activation is True]
quiescent = [cell for cell in all_cells if cell.activation is False]

active, filtered = filter_cells(active, 0.1)

for cell in filtered:
    visualizer.visualize_array(cell.image, "{}_cell{}".format(cell.filename, str(cell.value)))
    
quiescent, filtered = filter_cells(quiescent, 0.1)

for cell in filtered:
    visualizer.visualize_array(cell.image, "{}_cell{}".format(cell.filename, str(cell.value)))
print("finish filtering...")
# split into test/train
random.seed(10)
random.shuffle(active)
random.shuffle(quiescent)

active_split_ind = math.floor((len(active) / 5))
quiescent_split_ind = math.floor((len(quiescent) / 5))

test = active[:active_split_ind] + quiescent[:quiescent_split_ind]
train = active[active_split_ind:] + quiescent[quiescent_split_ind:]

# augment
print("start augmenting...")
augment_cells(test)
augment_cells(train)
print("end augmenting...")    

# save to folders
print("start saving...")
save_cells(test, "Images/Test")
save_cells(train, "Images/Train")
print("completed")