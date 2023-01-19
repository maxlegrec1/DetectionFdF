import numpy as np
import tensorflow as tf
from IA_conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from IA_dataset_for_nn import create_data_list

def unison_shuffled_copies(couple):     #mélange les données pour éviter le sur-apprentissage
    (a, b) = couple
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return (a[p], b[p])

def train():                #création du modèle

  (train_X, train_y)= unison_shuffled_copies(create_data_list('Training'))
  (val_X, val_Y) = create_data_list('Validation')

  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.MaxPooling2D(input_shape = (210,465,3)))
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

  history = model.fit(x = train_X, y = train_y, epochs = 3, batch_size = 128, validation_data = (val_X, val_Y))

  model.save("Saved_models/model.h5")
  return model

def predic(signals, model):     #prédiction des données
    predic = model(signals)

    L = [0]*len(predic)

    for i in range (len(predic)):
        L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))
    return L

def predic_total_signal(predics, fire_rate = 1/3):    #prédiction du signal total à partir des prédictions de ses enregistrements coupés

  s = 0  
  for i in range(len(predics)):
      s += predics[i][0]
  return "a fire" if s > fire_rate*len(predics) else 'not a fire'

def test():    #test du modèle
  (test_X, test_y) = create_data_list("Test")

  L = predic(test_X, train())

  cm = confusion_matrix(test_y, L)

  class_names = ['Negatif', 'Positif']

  show_confusion_matrix(cm, class_names)

if __name__ == "__main__":
  test()