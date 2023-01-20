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
    '''
    chargement du fichier audio

    Parameters
    ----------
    path : string
        -- chemin du fichier audio.
    
    Returns
    -------
    sig : array
        -- signal audio.
    '''
    sig, sr = sf.read(path)
    return (sig, sr)

def prepa(sig, sr):
    '''
    découpe du son original en sous sons de 4 secondes chacun
    
    Parameters
    ----------
    sig : array
        -- signal audio.
    sr : int
        -- fréquence d'échantillonnage.
    
    Returns
    -------

    list_of_sounds : list
        -- liste des sous sons.

    '''
    if len(sig.shape)>1:
        (i, j) = silence(sig[:,0])
        list_of_sounds=diviser_son(sig[i:j+1, 0], sr, 4)
    else:
        (i, j) = silence(sig[:])
        print(i,j)
        list_of_sounds=diviser_son(sig[i:j+1], sr, 4)
    return list_of_sounds



def spect(sig, sr, i):      #
    '''
    création du spectrogramme du signal audio et sauvegarde de l'image dans le dossier Main_specs sous le nom image_i.png

    Parameters
    ----------
    sig : array
        -- signal audio.
    sr : int
        -- fréquence d'échantillonnage.

    Returns
    -------
    None. (sauvegarde de l'image dans le dossier Main_specs sous le nom image_i.png)

    '''
    plotstft(sig, sr, plotpath='Main_specs/image_'+str(i)+'.png')

def resize(i):    
    '''
    redimensionnement de l'image du spectrogramme pour ne garder que la partie du spectrogramme correspondant aux fréquences qui nous intéressent (critère de Shannon)

    Parameters
    ----------
    i : int
        -- numéro de l'image dans le dossier Main_specs.
    
    Returns
    -------
    np.asarray(Image.open(path)) : array
        -- array de l'image redimensionnée.
    
    '''
    path = 'Main_specs/image_'+str(i)+'.png'
    resize_path(path)
    return np.asarray(Image.open(path))

def load_model(model_choice):     
    '''
    Write description here

    Parameters
    ----------
    model_choice : string
        -- choice of the model ("SVM" or "CNN")

    Returns
    -------
    model : model (SVM or CNN object)
        -- model choisi
    
    '''
    if model_choice == "CNN":
        model = tf.keras.models.load_model("Saved_models/CNN4.h5")
    elif model_choice == "SVM":
        model = pickle.load(open('SVM_1.sav','rb'))
    else:
        raise ValueError("Le choix de modèle n'est pas correct")
    return model



def pipeline(path_of_the_file, model_choice):
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
    #chargement du fichier audio
    (sound, sr) = load_file(path_of_the_file)
    
    #découpe du son original en sous sons de 4 secondes chacun
    prep_sounds = prepa(sound, sr)

    if model_choice == "CNN":

        specs, pred = [], []
        #boucle sur les sons de 4 secondes
        for i in range(len(prep_sounds)):

            #création de spectrogrammes à partir de transformées de fourier à fenêtres courtes
            spect(prep_sounds[i], sr, i)

            #crop du spectrogramme pour enlever les axes et les hautes fréquences
            specs.append(resize(i))

            #prédiction du modèle
            pred.append(predic(specs[i].reshape(1, 201, 462, 3), load_model(model_choice)))
        return predic_total_signal(pred)
    
    
    elif model_choice == "SVM":

        predicts=[]
        #boucle sur les sons de 4 secondes
        for i in range(len(prep_sounds)):
            
            #enregistrement des sons de 4 secondes dans un dossier temporaire
            sf.write("temp/"+str(i)+".wav", prep_sounds[i], sr)
            
            #prédiction du modèle
            predicts.append(predic_svm("temp/"+str(i)+".wav", load_model(model_choice)))
        
        #suppression des sons de 4 secondes du dossier temporaire
        for i in range(len(prep_sounds)):
            os.remove("temp/"+str(i)+".wav")
        return predic_total_signal(predicts)    

def main():
    '''
    execution de la pipeline

    Parameters
    ----------
    None -- prend en argument le modèle choisi et le path du fichier

    Returns
    -------
    None -- ouvre une fenêtre avec la prédiction
    '''
    model_choice = sys.argv[1]
    path_of_the_file = sys.argv[2]
    prediction = pipeline(path_of_the_file, model_choice)

    #ouvre la fenêtre avec la prédiction
    ctypes.windll.user32.MessageBoxW(0, "This audio sample is " + prediction, "Prediction", 0)
    
    return

if __name__ == "__main__":  
    main()


