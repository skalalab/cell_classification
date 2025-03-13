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
        plt.title("SDT")
        plt.show()
        
    elif data.ndim == 4:
        for i in range(data.shape[0]):
            plt.imshow(np.sum(data[i], axis = 2))
            plt.title("SDT {}".format(str(i)))
            plt.show()
            
            
# # visualize cells
# def visualize_cells(folder):
#     for cell in glob(folder + "/*.tif"):
#         with tiff.TiffFile(cell) as tif:
#             image = tif.asarray()
        
#         plt.imshow(np.sum(image, axis=2))
#         plt.title(Path(cell).name[:-4])
#         plt.show()
        