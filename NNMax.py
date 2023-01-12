#write a CNN model that takes for input a spectrogram and outputs a probability of the signal being a fire

#imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#data augmentation

#input shape = (210,465,3)
#training database size = 5000 spectrograms
#add dropout layers
#model
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(210, 465, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

#compile
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#data
import numpy as np
def unison_shuffled_copies(couple):     #mélange les données pour éviter le sur-apprentissage
    (a, b) = couple
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return (a[p], b[p])

               #création du modèle
from dataset_for_nn import create_data_list


(train_X, train_y)= unison_shuffled_copies(create_data_list('Training'))
print("train_X",len(train_X))
(val_X, val_Y) = create_data_list('Validation')
print("val_X",len(val_X))

#train model and graph the accuracy on the training and validation sets
history = model.fit(train_X, train_y, epochs=10, validation_data=(val_X, val_Y))

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#save model
model.save('NNMax.h5')




