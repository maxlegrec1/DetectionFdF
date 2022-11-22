from load_db import generate_arrays_from_file,load_dataset
import tensorflow as tf
#from tensorflow.keras import datasets, layers, models   
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def create_and_save_model():
    #function that loads dataset
    dataset_path_train="Dataset_spec/Training"
    batch_size=40
    batch_number=7000//batch_size #correspond a un batch size de 4000/800 = 3
    gen=generate_arrays_from_file(dataset_path_train,batch_number)
    print("KFJEZIOFZO",next(gen))

    #load model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (289,465,3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (289,465,3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (289,465,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    model.summary()
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])


    #load test dataset

    #print(test_X,test_y)
    #train model using generator, 20 epoch, print test_accuracy every batch
    model.fit(gen,epochs=1,steps_per_epoch=batch_number)
    #print test accuracy

    #because data is redundent in dataset, overfitting starts before 1 epoch




    #save model
    model.save("model.h5")



#load dataset_test
dataset_path_test="Dataset_spec/Test"
number_of_images_in_batch=700
dataset_size=700
nb_batch=dataset_size//number_of_images_in_batch
test_X,test_y=load_dataset(dataset_path_test,nb_batch,0)


#load model
model = tf.keras.models.load_model("model.h5")

test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
print("test_accccc",test_acc)


#prediction of model on test_X, feed the model with batches of 100 images
#store prediction in a list
predic=np.empty((0,2),int)
for i in range(0,len(test_X),100):
    print(i)
    incr_predict=model.predict(test_X[i:i+100])
    print(predic.shape)
    predic=np.append(predic,incr_predict,axis=0)



print(predic.shape)
L = [0]*len(predic)
for i in range (len(predic)):
    L[i] = list(predic[i]).index(max(predic[i]))
L=np.array(L)
Y = [0]*len(test_y)
for i in range (len(test_y)):
    Y[i] = list(test_y[i]).index(max(test_y[i]))
print(Y,L)
cm = confusion_matrix(Y, L)
print(cm)

#plot confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(cm, range(2),range(2))
plt.figure(figsize = (2,2))
sn.set(font_scale=1.4) #for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}) # font size
plt.show()
