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

# split sdt into single cells
def split_sdt(sdt, mask, output): 
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
    
    # single out each cell
    cell_images = []
    for cell_value in cell_values:
        # mask each cell
        cell_image = np.copy(sdt_data)
        for row in range(mask_data.shape[0]):
            for col in range(mask_data.shape[1]):
                if mask_data[row][col] != cell_value:
                    cell_image[row][col][:] = 0
                
        cell_images.append((cell_value, cell_image))
        

    
    return cell_images
    

def __find_bound__(upper, vertical):
    pass

# crop cell images
def crop_cells(cells):
    for cell in cells:
        # find bounds
        left = __find_bound__(upper=False, vertical=False)
        right = __find_bound__(upper=True, vertical=False)
        lower = __find_bound__(upper=False, vertical=True)
        upper = __find_bound__(upper=True, vertical=True)

        # crop
        cell = cell[lower:upper][left:right]

    return cells


# pad cells
def pad_cells(cells):
    pass


# save cells to folder
def save_cells(cells, output):
    for cell in cells:
        tiff.imwrite(output + "cell{}.tif".format(str(cell[0])), cell[1])

# run
output_path = "Images/" 
if not os.path.exists(output_path):
    os.makedirs(output_path)   
    
cells = split_sdt("Tcells-001.sdt", "Cell_mask_01.tif", output_path)
visualizer.visualize_cells(cells)