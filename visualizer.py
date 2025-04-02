# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:24:18 2025

@author: chris
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sdt_reader as sdt
import tifffile as tiff
from pathlib import Path

# intensity graph of sdt file
#
# param: file is path of sdt file
def visualize_sdt(file):
    data = sdt.read_sdt150(file)
    
    # visualize
    if data.ndim == 3:
        plt.imshow(np.sum(data, axis = 2)) 
        plt.title(file)
        plt.show()
        
    elif data.ndim == 4:
        for i in range(data.shape[0]):
            plt.imshow(np.sum(data[i], axis = 2))
            plt.title("{}: {}".format(file, str(i)))
            plt.show()
            
    return data[0]
            
# visualize tif
def visualize_tif(file):
    with tiff.TiffFile(file) as tif:
        image = tif.asarray()
    
    plt.imshow(np.sum(image, axis=2))
    plt.title(file)
    plt.show()
    

# visualize numpy array
def visualize_array(array, title, need_sum=True):
    if need_sum:
        array = np.sum(array, 2)
        
    plt.imshow(array)
    plt.title(title)
    plt.show()
    

visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_no.tif")
visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_fh.tif")
visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_fv.tif")
visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_r90.tif")
visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_r180.tif")
visualize_tif("C:/Users/chris/cell_classification/Images/Train/d4_active_cell21_r270.tif")


        
        