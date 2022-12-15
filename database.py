import os 
import soundfile as sf
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np
import wavio


sample_freq=44100
audio_duration=4

def trier_bonne_longueur(x): #on supprime les sons ne faisant pas 4 secondes
    folder="audio/fold"+str(x)
    folder2="audio/poubelle"
    for count, filename in enumerate(os.listdir(folder)):
            audio,samplerates=sf.read(f"{folder}/{filename}")
            temps=len(audio)/samplerates
            if temps!=4:
                print("c nul")
                src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)

def rangement(x): #on trie les fichiers ne contenant pas de feu par classes
    folder="audio/fold"+str(x)
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder2}{int(fichier[1])+1}/{filename}"
        os.rename(src,dst)
        print(int(fichier[1]))

def rennomer_autre(x): #on ajoute "Autre" devant chaque fichier
    folder="audio/fold"+str(x)
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}"
        dst =f"{folder}/Autre{filename}"
        os.rename(src,dst)

def distribution(x): #pour connaître la répartition selon les classes 
    folder="audio/fold"+str(x)
    liste=[0,0,0,0,0,0,0,0,0,0]
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        liste[int(fichier[1])]+=1
    print(liste)

def nbre_bonne_freq(): #on regarde combien de sons disposent de la fréquence d'échantillonage souhaité
    choix=[0 for k in range(10)]
    for k in range(1,11):
        folder="audio/fold"+str(k)
        print(k)
        nbre=0
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            temps=len(data)/samplerate
            tot+=1
            if samplerate==sample_freq:
                nbre+=1
        choix[k-1]=nbre
    print(choix)

def tri_bonne_freq(): #on ne garde que les fichiers avec une fréquence d'échantillonage de 44100 
    for k in range(1,11):
        folder="audio/fold"+str(k)
        folder2="audio/poubelle_freq"
        print(k)
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            if samplerate!=sample_freq:
                print("c nul")
                src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)

def silence(data):                  #enlever les silences de début et de fin
    i = 0
    while i < len(data) and data[i] == 0:
        i +=1
    j = len(data)-1
    while j > 0 and data[j] == 0:
        j -= 1

    return (i, j)

def diviser_son(data, sr = sample_freq, audio_duration = audio_duration): #diviser un son trop long en plusieurs sons plus petits
    nombre_points = len(data)
    points_esperes = sample_freq * audio_duration
    if nombre_points >= points_esperes:
        audio1 = data[:points_esperes]
        return [audio1] + diviser_son(data[points_esperes:])
    else:
        return []

def feu_to_dataset(): #on transfere les sons contenant du feu dans le dataset (en les divisant en sons plus petits)
    for count, filename in enumerate(os.listdir("Feu")):
        print(count)
        data,samplerate=sf.read(f"Feu/{filename}")
        (i,j)=silence(data[:,0])
        signaux=diviser_son(data[i:j])
        for k in range(len(signaux)):
            wavio.write(f"Dataset/{filename[:-4]}_{k}.wav", signaux[k], samplerate, sampwidth=4)

choix_final=[400, 175, 567, 435, 400, 400, 16, 400, 400, 400]

def trier_mono(): #on enleve les fichiers qui ne sont pas en stereo
    folder2="audio/poubelle_mono"
    for k in range(1,11):
        folder="audio/fold"+str(k)
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            try:
                print(data.shape[1])
            except:
                src =f"{folder}/{filename}" 
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)

stereos=[640, 175, 559, 408, 468, 450, 16, 454, 403, 654] #nombre de fichiers en stéréo par classes

def vider_poubelle(poubelle): #si on souhaite récuperer des fichiers
    folder="audio/"+poubelle
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder2}{int(fichier[1])+1}/{filename}"
        os.rename(src,dst)

def autre_to_dataset(liste): #on transfère les fichiers ne contenant pas de feu dans le dataset
    #on prend en entrée une liste qui nous donne le nombre de sons à prendre par classe
    folder2="Dataset"
    for k in range(1,11):
        nbre=liste[k-1]
        folder="audio/fold"+str(k)
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            if count<=nbre:
                src =f"{folder}/{filename}" 
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)
            else:
                break

choix2=[400,175,559,408,400,400,16,400,400,400]

def compter_dataset(): #pour connaître la proportion feu/pas feu
    folder="Dataset"
    feu=0
    autre=0
    for count, filename in enumerate(os.listdir(folder)):
        if filename[:3]=="Feu":
            feu+=1
        elif filename[:3]=="Aut":
            autre+=1
        else:
            print('il y a un probleme avec')
            print(filename)
    print('Il y a '+str(feu)+" feu")
    print("Il y a "+str(autre)+" autre")

def dataset_to_autre(): #Pour enlever les fichiers ne contenant pas de feu
    print("vide le dataset")
    folder="Dataset/Validation"
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        print("boucle")
        if filename[:3]=="Aut":
            print("if")
            fichier=filename.split('-',3)
            src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
            dst =f"{folder2}{int(fichier[1])+1}/{filename}"
            os.rename(src,dst)

def delete_feu_dataset(): #on supprime les feux du dataset
    folder="Dataset/Validation"
    for count, filename in enumerate(os.listdir(folder)):
        if filename[:3]=="Feu":
            os.remove(f"{folder}/{filename}")

def clear_dataset(): #on vide entièrement le dataset
    dataset_to_autre()
    delete_feu_dataset()


def check_dataset(folder,freq_echant,temps):
    print("Début de check_dataset")
    sil,freq,tem,feu,autre,nom,ext,mono=0,0,0,0,0,0,0,0
    for count, filename in enumerate(os.listdir(folder)):
        data,samplerate=sf.read(f"{folder}/{filename}")
        if silence(data[:,0])!=(0,len(data)-1): #vérifie les silences
            sil+=1
        if samplerate!=freq_echant: #vérifie la fréquence d'échantillonnage
            freq+=1
        if len(data)!=temps*freq_echant: #vérifie le temps des enregistrements
            tem+=1
        if filename[:3]=="Feu": #compte le nombre de feux
            feu+=1
        if filename[:3]=="Aut": #vérifie le nombre de non feux
            autre+=1
        if filename[:3]!="Feu" and filename[:3]!="Aut": #vérifie le nom des fichiers
            nom+=1
        if filename[-4:]!=".wav": #vérifie l'extension des fichiers
            ext+=1
        try:
            data.shape[1]
        except:
            mono+=1
    if mono!=0:
        print("Il y a "+str(mono)+" fichiers mono")
    if ext!=0:
        print("Il y a "+str(ext)+" fichiers qui ne finissent pas en .wav")
    if nom!=0:
        print("Il y a "+str(nom)+" fichiers qui ne sont pas rennommés correctement")
    if tem!=0:
        print("Il y a "+str(tem)+" sons qui n'ont pas la bonne durée")
    if freq!=0:
        print("Il y a "+str(freq)+" sons qui ne sont pas à la bonne fréquence d'échantillonnage")
    if sil!=0:
        print("Il y a "+str(sil)+" sons contenants des silences au début ou à la fin")
    print("La proportion feu/pas feu est de "+str(feu)+"/"+str(autre))
    print("fin")

#check_dataset("DetectionFdF/Dataset",44100,4)


def training_val_test():
    Feu,Autre=3455,3564
    folder="Dataset"
    list=["Training","Validation","Test"]
    coeff=[0.70,0.20,0.10]
    for i in range(3):
        print(i)
        f,a=0,0
        for count, filename in enumerate(os.listdir(folder)):
            if filename[:3]=="Feu" and f<coeff[i]*Feu:
                src =f"{folder}/{filename}" 
                dst =f"{folder}/{list[i]}/{filename}"
                os.rename(src,dst)
                f+=1
            if filename[:3]=="Aut" and a<coeff[i]*Autre:
                src =f"{folder}/{filename}" 
                dst =f"{folder}/{list[i]}/{filename}"
                os.rename(src,dst)
                a+=1
            if f>coeff[i]*Feu and a>coeff[i]*Autre:
                break
        


def vider_training_val_test():
    folder="Dataset"
    list=["Training","Validation","Test"]
    for i in range(3):
        for count, filename in enumerate(os.listdir(f"{folder}/{list[i]}")):
            src =f"{folder}/{list[i]}/{filename}"  
            dst =f"{folder}/{filename}"
            os.rename(src,dst)


def autre_to_dataset2(liste): #on transfère les fichiers ne contenant pas de feu dans le dataset
    #on prend en entrée une liste qui nous donne le nombre de sons à prendre par classe
    folder2="Dataset"
    for k in range(1,11):
        nbre=liste[k-1]
        folder="audio/fold"+str(k)
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            if count<=nbre*0.7:
                src =f"{folder}/{filename}" 
                dst =f"{folder2}/Training/{filename}"
                os.rename(src,dst)
            elif count<=nbre*0.9:
                src =f"{folder}/{filename}" 
                dst =f"{folder2}/Validation/{filename}"
                os.rename(src,dst)
            elif count<=nbre:
                src =f"{folder}/{filename}" 
                dst =f"{folder2}/Test/{filename}"
                os.rename(src,dst)
            else:
                break
choix2=[400,175,559,408,400,400,16,400,400,400]


def feu_to_dataset2(): #on transfère les sons contenant du feu dans le dataset (en les divisant en sons plus petits)
    for count, filename in enumerate(os.listdir("Feu")):
        name=filename.split('_',2)
        if int(name[1][:-4])<=50:
            folder="Dataset/Training"
        elif int(name[1][:-4])<=65:
            folder="Dataset/Validation"
        else:
            folder="Dataset/Test"
        data,samplerate=sf.read(f"Feu/{filename}")
        (i,j)=silence(data[:,0])
        signaux=diviser_son(data[i:j])
        for k in range(len(signaux)):
            wavio.write(f"{folder}/{filename[:-4]}_{k}.wav", signaux[k], samplerate, sampwidth=4)

