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
        cell_images.append([cell_value, cropped_image, cell_image])
    
    return cell_images


# pad cells
def pad_cells(cells):
    # get largest dimension as size
    cells.sort(key=lambda x: max(x[1].shape[0], x[1].shape[1]), reverse=True)
    size = max(cells[0][1].shape[0], cells[0][1].shape[1])
    
    # make size even
    if size % 2 != 0:
        size += 1
    
    # pad all images to be size x size
    for cell in cells:
        vert_padding = size - cell[1].shape[0]
        side_padding = size - cell[1].shape[1]
        cell[1] = np.pad(cell[1], ((math.floor(vert_padding/2),math.ceil(vert_padding/2)), (math.floor(side_padding/2),math.ceil(side_padding/2)), (0, 0)), 'constant')   
    


# # save cells to folder
def save_cells(cells, output):
    for cell in cells:
        tiff.imwrite(output + "cell{}.tif".format(str(cell[0])), cell[1])

# run
output_path = "Images/" 
if not os.path.exists(output_path):
    os.makedirs(output_path)   
    
cells = split_sdt("Tcells-001.sdt", "Cell_mask_01.tif")
pad_cells(cells)
visualizer.visualize_cells(cells)