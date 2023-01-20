import tensorflow as tf
from IA_dataset_for_nn import create_data_list
from IA_conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np

#charge le modèle
model = tf.keras.models.load_model("Saved_models/model.h5")

#test du modèle
(test_X, test_y) = create_data_list("Test")

predic = model(test_X)

L = [0]*len(predic)

for i in range (len(predic)):
  L[i] = list(np.array(predic[i])).index(max(np.array(predic[i])))

cm = confusion_matrix(test_y, L)

class_names = ['Négatif', 'Positif']

show_confusion_matrix(cm, class_names)