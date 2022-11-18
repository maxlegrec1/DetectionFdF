# import the modules
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 
# get the path/directory
def create_data_list(folder):
    folder_dir = "Dataset_spec/"+folder
    data_list, label_list = [],[]
    for images in os.listdir(folder_dir):
        data_list.append(Image.open("Dataset_spec/"+folder+"/"+images))
        # check if the image ends with png
        if (images.startswith("Autre")):
            label_list.append(0)
        else:
            label_list.append(1)

    return (data_list, label_list)


        
