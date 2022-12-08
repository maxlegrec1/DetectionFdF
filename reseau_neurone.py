import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from dataset_for_nn import create_data_list

def unison_shuffled_copies(couple):
    (a, b) = couple
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return (a[p], b[p])

def train():

  (train_X, train_y)= unison_shuffled_copies(create_data_list('Training'))
  (val_X, val_Y) = create_data_list('Validation')

  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.MaxPooling2D(input_shape = (289,465,3)))
  model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(16, 3, activation = 'relu'))
  model.add(tf.keras.layers.MaxPooling2D())

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(64, activation = 'relu'))
  model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

  model.summary()

  model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

  history = model.fit(x = train_X, y = train_y, epochs = 1, batch_size = 128, validation_data = (val_X, val_Y))

  """plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label='val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.legend(loc='lower right')
  plt.show()"""

  model.save("Saved_models/model.h5")
  return model

def predic(signals, model):
    predic = model(signals)

    L = [0]*len(predic)

    for i in range (len(predic)):
        L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))
    return L

def predic_total_signal(predics, fire_rate = 1/3):

  s = 0  
  for i in range(len(predics)):
      s += predics[i][0]
  return "Fire" if s > fire_rate*len(predics) else 'Not fire'

def test():
  (test_X, test_y) = create_data_list("Test")

  L = predic(test_X, train())

  cm = confusion_matrix(test_y, L)

  class_names = ['Negatif', 'Positif']

  show_confusion_matrix(cm, class_names)

if __name__ == "__main__":
  test()