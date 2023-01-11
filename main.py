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
import sys
import ctypes
import pickle
from SVM import predic_svm
import os

def load_file(path):    #chargement du fichier audio
    sig, sr = sf.read(path)
    return (sig, sr)

def prepa(sig, sr):
    #print(sig.shape)     #préparation du signal pour répondre au format souhaité
    if len(sig.shape)>1:
        (i, j) = silence(sig[:,0])
        #print(i,j)
        list_of_sounds=diviser_son(sig[i:j+1, 0], sr, 4)
    else:
        (i, j) = silence(sig[:])
        print(i,j)
        list_of_sounds=diviser_son(sig[i:j+1], sr, 4)
    return list_of_sounds

def filtrage(sig):      #filtrage du signal
    sig = filtre(sig)
    return sig

def spect(sig, sr, i):      #création du spectrogramme
    plotstft(sig, sr, plotpath='Main_specs/image_'+str(i)+'.png')

def resize(i):    #redimensionnement du spectrogramme
    path = 'Main_specs/image_'+str(i)+'.png'
    resize_path(path)
    return np.asarray(Image.open(path))

def load_model(model_choice):       #chargement du modèle
    if model_choice == "CNN":
        model = tf.keras.models.load_model("Saved_models/model2.h5")
    elif model_choice == "SVM":
        model = pickle.load(open('SVM_1.sav','rb'))
    else:
        raise ValueError("Le choix de modèle n'est pas correct")
    return model

if __name__ == "__main__":  
    
    model_choice = sys.argv[1]
    path_of_the_file = sys.argv[2]
    (sound, sr) = load_file(path_of_the_file)
    prep_sounds = prepa(sound, sr)
    if model_choice == "CNN":
        specs, pred = [], []
        for i in range(len(prep_sounds)):
            filtered = filtrage(prep_sounds[i])
            spect(filtered, sr, i)
            specs.append(resize(i))
            pred.append(predic(specs[i].reshape(1, 210, 465, 3), load_model(model_choice)))

        ctypes.windll.user32.MessageBoxW(0, "This audio sample is " + predic_total_signal(pred), "Prediction", 0)
    elif model_choice == "SVM":
        predicts=[]
        for i in range(len(prep_sounds)):
            filtered = filtrage(prep_sounds[i])
            #write the filtered sound as the "ith" wav in the temp folder
            sf.write("temp/"+str(i)+".wav", filtered, sr)
            predicts.append(predic_svm("temp/"+str(i)+".wav", load_model(model_choice)))
        #clear the temp folder
        for i in range(len(prep_sounds)):
            os.remove("temp/"+str(i)+".wav")
        ctypes.windll.user32.MessageBoxW(0, "This audio sample is " + predic_total_signal(predicts), "Prediction", 0)

def execute(path_of_the_file, model_choice):
    if model_choice == "CNN":
        (sound, sr) = load_file(path_of_the_file)
        prep_sounds = prepa(sound, sr)
        specs, pred = [], []
        for i in range(len(prep_sounds)):
            filtered = filtrage(prep_sounds[i])
            spect(filtered, sr, i)
            specs.append(resize(i))
            pred.append(predic(specs[i].reshape(1, 210, 465, 3), load_model(model_choice)))
        return predic_total_signal(pred)
    elif model_choice == "SVM":
        (sound, sr) = load_file(path_of_the_file)
        prep_sounds = prepa(sound, sr)
        predicts=[]
        for i in range(len(prep_sounds)):
            filtered = filtrage(prep_sounds[i])
            #write the filtered sound as the "ith" wav in the temp folder
            sf.write("temp/"+str(i)+".wav", filtered, sr)
            predicts.append(predic_svm("temp/"+str(i)+".wav", load_model(model_choice)))
        #clear the temp folder
        for i in range(len(prep_sounds)):
            os.remove("temp/"+str(i)+".wav")
        return predic_total_signal(predicts)

