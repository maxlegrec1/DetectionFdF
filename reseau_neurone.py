from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from load_db import load_dataset
from load_db import generate_arrays_from_file
#(train_X, train_y), (test_X, test_y) = mnist.load_data()
nb_batch = 300


#print(train_X)
'''print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))'''


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


(test_X,test_y) = load_dataset("Dataset_spec/Test",7,0)
batch_number=800 #correspond a un batch size de 4000/800 = 3
gen=generate_arrays_from_file("Dataset_spec/Training",batch_number)


model.fit_generator(gen,batch_number,10,validation_data=(test_X,test_y))

test_X=test_X[:100]
test_y=test_y[:100]
test_y=np.argmax(test_y, axis=1)
predic = model(test_X)

print(np.array(predic[0]))

L = [0]*len(predic)

for i in range (len(predic)):
  L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))

cm = confusion_matrix(test_y, L)
print(cm)
class_names = [str(i) for i in range (10)]

show_confusion_matrix(cm, class_names)