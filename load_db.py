import numpy as np
from PIL import Image
import os
from tqdm import tqdm
def load_dataset(DB_PATH,nb_batch=1,current_batch_number=0):
    X=np.empty((0,289,465,3),int)
    Y=np.empty((0,2),int)
    directory=os.fsencode(DB_PATH)
    list=os.listdir(directory)
    np.random.shuffle(list)
    batch_size=len(list)//nb_batch
    list_shrunk=list[current_batch_number*batch_size:(current_batch_number+1)*batch_size]
    for spec_path in tqdm(list_shrunk):
        file_name=os.fsdecode(spec_path)
        spec=np.asarray(Image.open(DB_PATH+"/"+file_name))
        spec=np.expand_dims(spec,axis=0)
        X=np.append(X,spec,axis=0)
        #print(Y)
        if file_name[0]=='A': 
            Y=np.append(Y,np.expand_dims(np.array([1,0]),axis=0),axis=0)
        else:
            Y=np.append(Y,np.expand_dims(np.array([0,1]),axis=0),axis=0)
    print("done")
    return (X,Y)
#X,Y=load_dataset("Dataset_spec/Training",100,50)
#print(Y)
def generate_arrays_from_file(INPUT_PATH,batch_number):
    directory=os.fsencode(INPUT_PATH)
    list=os.listdir(directory)
    np.random.shuffle(list)
    i=0
    c=0
    while 1:
        X=np.empty((0,289,465,3),int)
        Y=np.empty((0,2),int)
        batch_size=len(list)//batch_number
        list_shrunk=list[i*batch_size:(i+1)*batch_size]
        for spec_path in list_shrunk:
            file_name=os.fsdecode(spec_path)
            spec=np.asarray(Image.open(INPUT_PATH+"/"+file_name))
            spec=np.expand_dims(spec,axis=0)
            X=np.append(X,spec,axis=0)
            #print(Y)
            if file_name[0]=='A': 
                Y=np.append(Y,np.expand_dims(np.array([1,0]),axis=0),axis=0)
            else:
                Y=np.append(Y,np.expand_dims(np.array([0,1]),axis=0),axis=0)
        #print(X)
        yield (X,Y)
        i+=1
        c+=1
        if i==batch_number:
            i=0