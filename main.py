import tensorflow as tf
import soundfile as sf
import numpy as np
from PIL import Image
import sys
import ctypes
import pickle
import os
#nos fichiers (à renommer ai_xxxx.py) ----------------------------------------------------------------------------
from IA_database import silence, diviser_son
from IA_spectrogramme import plotstft
from IA_resize_db import resize_path
from IA_reseau_neurone import predic
from IA_reseau_neurone import predic_total_signal
from IA_SVM import predic_svm
#-------------------------------------------------------------------------------------------


def load_file(path):    
    '''chargement du fichier audio*
    '''
    sig, sr = sf.read(path)
    return (sig, sr)

def prepa(sig, sr):
    '''découpe du son original en sous sons de 4 secondes chacun'''
    if len(sig.shape)>1:
        (i, j) = silence(sig[:,0])
        list_of_sounds=diviser_son(sig[i:j+1, 0], sr, 4)
    else:
        (i, j) = silence(sig[:])
        print(i,j)
        list_of_sounds=diviser_son(sig[i:j+1], sr, 4)
    return list_of_sounds



def spect(sig, sr, i):      #
    '''création de spectrogrammes à partir de transformées de fourier à fenêtres courtes'''
    plotstft(sig, sr, plotpath='Main_specs/image_'+str(i)+'.png')

def resize(i):    
    '''crop du spectrogramme pour enlever les axes et les hautes fréquences'''
    path = 'Main_specs/image_'+str(i)+'.png'
    resize_path(path)
    return np.asarray(Image.open(path))

def load_model(model_choice):     
    '''chargement du modèle en fonction du modèle choisi'''
    if model_choice == "CNN":
        model = tf.keras.models.load_model("Saved_models/model.h5")
    elif model_choice == "SVM":
        model = pickle.load(open('SVM_1.sav','rb'))
    else:
        raise ValueError("Le choix de modèle n'est pas correct")
    return model



def execute(path_of_the_file, model_choice):
    '''
    execution de la pipeline sur un fichier précis, sans passer par la console

    Parameters
    ----------
    path_of_the_file : string
        -- path of the file
    model_choice : string
        -- choice of the model ("SVM" or "CNN")

    Returns
    -------
    prediction : string
        -- prediction du model ("a fire" or "not a fire")
    '''
    if model_choice == "CNN":
        (sound, sr) = load_file(path_of_the_file)
        prep_sounds = prepa(sound, sr)
        specs, pred = [], []
        #boucle sur les sons de 4 secondes
        for i in range(len(prep_sounds)):
            filtered = prep_sounds[i]
            spect(filtered, sr, i)
            specs.append(resize(i))
            pred.append(predic(specs[i].reshape(1, 210, 465, 3), load_model(model_choice)))
        return predic_total_signal(pred)
    elif model_choice == "SVM":
        (sound, sr) = load_file(path_of_the_file)
        prep_sounds = prepa(sound, sr)
        predicts=[]
        for i in range(len(prep_sounds)):
            filtered = prep_sounds[i]
            #write the filtered sound as the "ith" wav in the temp folder
            sf.write("temp/"+str(i)+".wav", filtered, sr)
            predicts.append(predic_svm("temp/"+str(i)+".wav", load_model(model_choice)))
        #clear the temp folder
        for i in range(len(prep_sounds)):
            os.remove("temp/"+str(i)+".wav")
        return predic_total_signal(predicts)    

if __name__ == "__main__":  
    '''
    Execution de la pipeline depuis la console.

    INPUT : depuis le terminal
    Exemple ; !python3 main.py sound.wav CNN
    Le résultat est indiqué dans une nouvelle fenêtre qui s'ouvre.
    '''


    model_choice = sys.argv[1]
    path_of_the_file = sys.argv[2]
    (sound, sr) = load_file(path_of_the_file)
    prep_sounds = prepa(sound, sr)
    if model_choice == "CNN":
        specs, pred = [], []
        for i in range(len(prep_sounds)):
            filtered = prep_sounds[i]
            spect(filtered, sr, i)
            specs.append(resize(i))
            pred.append(predic(specs[i].reshape(1, 210, 465, 3), load_model(model_choice)))

        ctypes.windll.user32.MessageBoxW(0, "This audio sample is " + predic_total_signal(pred), "Prediction", 0)
    elif model_choice == "SVM":
        predicts=[]
        for i in range(len(prep_sounds)):
            filtered = prep_sounds[i]
            #write the filtered sound as the "ith" wav in the temp folder
            sf.write("temp/"+str(i)+".wav", filtered, sr)
            predicts.append(predic_svm("temp/"+str(i)+".wav", load_model(model_choice)))
        #clear the temp folder
        for i in range(len(prep_sounds)):
            os.remove("temp/"+str(i)+".wav")
        ctypes.windll.user32.MessageBoxW(0, "This audio sample is " + predic_total_signal(predicts), "Prediction", 0)



