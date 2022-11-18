from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(train_X[0].shape)
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, activation = 'relu', input_shape = (28,28,1)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

history = model.fit(train_X, train_y, epochs = 5, batch_size = 16, validation_data = (test_X, test_y))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

predic = model(test_X)

print(np.array(predic[0]))

L = [0]*len(predic)

for i in range (len(predic)):
  L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))

cm = confusion_matrix(test_y, L)

class_names = [str(i) for i in range (10)]

show_confusion_matrix(cm, class_names)