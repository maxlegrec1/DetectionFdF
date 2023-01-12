from main import execute
import os
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

DB_TEST = "Nature2"

if __name__ == "__main__":
    y_pred=[]
    y_true=[]
    for WAVFILE in tqdm(os.listdir(DB_TEST)):
        if WAVFILE.endswith(".wav"):
            pred=execute(DB_TEST + "/" + WAVFILE, "SVM")
            print(pred)
            bool=True
            if WAVFILE[:3]=="Feu":
                y_true.append(1)
            elif WAVFILE[:3]=="Aut":
                y_true.append(0)
            else:
                bool=False
                print('erreur')
            if bool:
                if pred=="a fire":
                    y_pred.append(1)
                else:
                    y_pred.append(0)
    print(len(y_pred), len(y_true))
    cm = confusion_matrix(y_true, y_pred,labels=[0,1])
    print(cm)
    class_names = ['Negatif', 'Positif']
    show_confusion_matrix(cm, class_names)