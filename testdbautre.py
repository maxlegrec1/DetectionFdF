from main import execute
import os

DB_TEST = "Dataset_Jagawana"

if __name__ == "__main__":
    counter=0
    for WAVFILE in os.listdir(DB_TEST):
        if WAVFILE.endswith(".wav"):
            print(WAVFILE)
            pred=execute(DB_TEST + "/" + WAVFILE, "CNN")
            print(pred)
            if pred=="Fire":
                counter+=1
    print("pourcentage de feux détectés : ", counter/len(os.listdir(DB_TEST)))