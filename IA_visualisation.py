from main import pipeline
import os
from IA_conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


#Evaluation du modèle sur un dataset

#Choix du dataset utilisé pour l'évaluation
DB_TEST = "Nature2"     

if __name__ == "__main__":
    y_pred=[]
    y_true=[]

    #iterations sur les fichiers du dataset
    for WAVFILE in tqdm(os.listdir(DB_TEST)):
        if WAVFILE.endswith(".wav"):

            #voir IA_loading_model.py pour comprendre le fonctionnement de pipeline
            pred=pipeline(DB_TEST + "/" + WAVFILE, "CNN")        
            print(pred)
            bool=True

            #création de la liste des vraies valeurs
            if WAVFILE[:3]=="Feu":
                y_true.append(1)
            elif WAVFILE[:3]=="Aut":
                y_true.append(0)
            else:
                bool=False
            if bool:

                #création de la liste des prédiction
                if pred=="a fire":
                    y_pred.append(1)
                else:
                    y_pred.append(0)
    
    #création de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred,labels=[0,1])
    
    class_names = ['Negatif', 'Positif']

    #affichage de la matrice de confusion
    show_confusion_matrix(cm, class_names)