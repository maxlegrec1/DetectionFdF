import numpy as np
from PIL import Image
import os
from tqdm import tqdm
def load_dataset(DB_PATH,nb_batch=1,current_batch_number=0):
    X=np.empty((0,645,1168,3),int)
    Y=np.array([])
    directory=os.fsencode(DB_PATH)
    list=os.listdir(directory)
    batch_size=len(list)//nb_batch
    list_shrunk=list[current_batch_number*batch_size:(current_batch_number+1)*batch_size]
    for spec_path in tqdm(list_shrunk):
        file_name=os.fsdecode(spec_path)
        spec=np.asarray(Image.open(DB_PATH+"/"+file_name))
        spec=np.expand_dims(spec,axis=0)
        X=np.append(X,spec,axis=0)

        if file_name[0]=='A':
            Y=np.append(Y,0)
        else:
            Y=np.append(Y,1)
    print("done")
    return (X,Y)
X,Y=load_dataset("Dataset_spec/Training",100,0)
