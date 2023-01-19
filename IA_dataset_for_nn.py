# import the modules
import os
from PIL import Image
import numpy as np
 
# get the path/directory
def create_data_list(folder):
    i = 0
    folder_dir = "Dataset_spec/"+folder
    data_list, label_list = [],[]
    for images in os.listdir(folder_dir):
        data_list.append(np.asarray(Image.open("Dataset_spec/"+folder+"/"+images)))
        # check if the image ends with png
        if (images.startswith("Autre")):
            label_list.append(0)
        else:
            label_list.append(1)
        i+=1
        if i % 500 == 0:
            print(i)

    label_array = np.array(label_list)
    print("label ok")
    data_array = np.array( data_list )

    print("data ok")

    return data_array, label_array
        
