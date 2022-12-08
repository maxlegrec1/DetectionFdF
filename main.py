import tensorflow as tf
import torchaudio
from database import silence, diviser_son
import numpy as np
from FILTRE import filtre
from spectrogramme import plotstft
from resize_db import resize_en_place
from reseau_neurone import predic

def load_file(path):
    sig, sr = torchaudio.load(path)
    return (sig, sr)

def prepa(sig, sr):
    (i, j) = silence(sig)
    list_of_sounds=diviser_son(sig[i:j])
    return list_of_sounds

def filtrage(sig):
    return filtre(sig)

def spect(sig, sr):
    return plotstft(sig, sr)

def resize(image):
    return resize_en_place(image)

def load_model(model_choice):
    if model_choice == "CNN":
        model = tf.keras.models.load_model("Saved_models/model.h5")
    elif model_choice == "SVM":
        raise ValueError("Pas encore implémenté")
    else:
        raise ValueError("Le choix de modèle n'est pas correct")
    return model

if __name__ == "__main__":
    
    model_choice = input("Choice of model : CNN or SVM")
    path_of_the_file = input("Path of the file")
    (sound, sr) = load_file(path_of_the_file)
    print(sound)
    prep_sounds = prepa(sound, sr)
    print(prep_sounds)

    specs = []
    for sound in prep_sounds:
        filtered = filtrage(sound)
        specs.append(resize(spect(filtered)))
    if len(specs) == 0:
        raise ValueError("Le fichier faisait moins de 4 secondes")
    pred = predic(specs, load_model(model_choice))

    print(pred)