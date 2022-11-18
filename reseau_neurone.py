from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from dataset_for_nn import create_data_list

(train_X, train_y), (val_X, val_Y), (test_X, test_y) = create_data_list('Training'), create_data_list('Validation'), create_data_list("Test")


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu', input_shape = (645,1168,3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu', input_shape = (645,1168,3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu', input_shape = (645,1168,3)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

history = model.fit(train_X, train_y, epochs = 1, batch_size = 128, validation_data = (val_X, val_Y))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

predic = model(test_X)

print(np.array(predic[0]))

"""L = [0]*len(predic)

for i in range (len(predic)):
  L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))

cm = confusion_matrix(test_y, L)

class_names = ['Pas feu', 'Feu']

show_confusion_matrix(cm, class_names)"""