import tensorflow as tf
import soundfile as sf
import cv2 as cv
import numpy as np
from PIL import Image
from database import silence, diviser_son
from FILTRE import filtre
from spectrogramme import plotstft
from resize_db import resize_path
from reseau_neurone import predic
from reseau_neurone import predic_total_signal

def load_file(path):    #chargement du fichier audio
    sig, sr = sf.read(path)
    return (sig, sr)

def prepa(sig, sr):     #préparation du signal pour répondre au format souhaité
    (i, j) = silence(sig[:,0])
    list_of_sounds=diviser_son(sig[i:j+1, 0], sr, 4)
    return list_of_sounds

def filtrage(sig):      #filtrage du signal
    return filtre(sig)

def spect(sig, sr, i):      #création du spectrogramme
    plotstft(sig, sr, plotpath='Main_specs/image_'+str(i)+'.png')

def resize(i):    #redimensionnement du spectrogramme
    path = 'Main_specs/image_'+str(i)+'.png'
    resize_path(path)
    return np.asarray(Image.open(path))

def load_model(model_choice):       #chargement du modèle
    if model_choice == "CNN":
        model = tf.keras.models.load_model("Saved_models/model.h5")
    elif model_choice == "SVM":
        raise ValueError("Pas encore implémenté")
    else:
        raise ValueError("Le choix de modèle n'est pas correct")
    return model

if __name__ == "__main__":  
    
    model_choice = input("Choice of model : CNN or SVM ")
    path_of_the_file = input("Path of the file ")
    (sound, sr) = load_file(path_of_the_file)
    prep_sounds = prepa(sound, sr)
    specs, pred = [], []
    for i in range(len(prep_sounds)):
        filtered = filtrage(prep_sounds[i])
        spect(filtered, sr, i)
        specs.append(resize(i))
        pred.append(predic(specs[i].reshape(1, 289, 465, 3), load_model(model_choice)))

    print(predic_total_signal(pred))