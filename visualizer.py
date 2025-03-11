# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:24:18 2025

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import sdt_reader as sdt

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
            
            
# visualize cells
def visualize_cells(cells):
    for cell in cells:
        plt.imshow(np.sum(cell[1], axis=2))
        plt.title("Cell_Cropped{}".format(str(cell[0])))
        plt.show()
        plt.imshow(np.sum(cell[2], axis=2))
        plt.title("Cell_Original{}".format(str(cell[0])))
        plt.show()