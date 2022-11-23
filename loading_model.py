import tensorflow as tf
from dataset_for_nn import create_data_list
from shuffle import unison_shuffled_copies
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np

model = tf.keras.models.load_model("model.h5")

(test_X, test_y) = create_data_list("Test")

predic = model(test_X)

L = [0]*len(predic)

for i in range (len(predic)):
  L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))

cm = confusion_matrix(test_y, L)

class_names = ['Pas feu', 'Feu']

show_confusion_matrix(cm, class_names)