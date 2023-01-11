from main import execute
import os
from conf_matrix import show_confusion_matrix
from sklearn.metrics import confusion_matrix


DB_TEST = "../Malange2"

if __name__ == "__main__":
    y_pred=[]
    y_true=[]
    for WAVFILE in os.listdir(DB_TEST):
        if WAVFILE.endswith(".wav"):
            pred=execute(DB_TEST + "/" + WAVFILE, "CNN")
            bool=True
            if WAVFILE[:3]=="Feu":
                y_true.append(1)
            elif WAVFILE[:3]=="Aut":
                y_true.append(0)
            else:
                 bool=False
            if bool:
                if pred=="Fire":
                    y_pred.append(1)
                else:
                    y_pred.append(0)
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Negatif', 'Positif']
    show_confusion_matrix(cm, class_names)