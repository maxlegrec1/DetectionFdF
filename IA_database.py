import os 
import soundfile as sf
import wavio


sample_freq=44100
audio_duration=4

def trier_bonne_longueur(x): #on supprime de la base de données les sons faisant moins de 4 secondes
    folder="audio/fold"+str(x)
    folder2="audio/poubelle"
    for _, filename in enumerate(os.listdir(folder)):
            audio,samplerates=sf.read(f"{folder}/{filename}")
            temps=len(audio)/samplerates
            if temps!=4:
                print("Erreur de temps")
                src =f"{folder}/{filename}"  
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)

def rangement(x): #on trie les fichiers ne contenant pas de feu par classes (il y 10 classes différentes)
    folder="audio/fold"+str(x)
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        src =f"{folder}/{filename}"  
        dst =f"{folder2}{int(fichier[1])+1}/{filename}"
        os.rename(src,dst)
        print(int(fichier[1]))

def renommer_autre(): #on ajoute "Autre" devant chaque fichier ne contenant pas de feu (pour simplifier le traitement ulterieur)
    folder="./Nature2"
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}"
        dst =f"{folder}/Autre{filename}"
        os.rename(src,dst)

def distribution(x): #pour connaître la répartition des sons selon les différentes classes 
    folder="audio/fold"+str(x)
    liste=[0,0,0,0,0,0,0,0,0,0]
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        liste[int(fichier[1])]+=1
    print(liste)

def nbre_bonne_freq(): #on regarde combien de sons disposent de la fréquence d'échantillonage souhaitée par classes
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

def tri_bonne_freq(): #on ne garde que les fichiers avec une fréquence d'échantillonage de 44 100 Hz (on transfère les autres dans un dossier poubelle)
    for k in range(1,11):
        folder="audio/fold"+str(k)
        folder2="audio/poubelle_freq"
        print(k)
        for count, filename in enumerate(os.listdir(folder)):
            data,samplerate=sf.read(f"{folder}/{filename}")
            if samplerate!=sample_freq:
                src =f"{folder}/{filename}"  
                dst =f"{folder2}/{filename}"
                os.rename(src,dst)

def silence(data): #on enleve les silences possiblement présent au début et à la fin de l'enregistrement
    i = 0
    while i < len(data) and data[i] == 0:
        i += 1
    j = len(data)-1
    while j > 0 and data[j] == 0:
        j -= 1

    return (i, j)

def diviser_son(data, sr = sample_freq, audio_duration = audio_duration): 
    '''
    On divise un son trop long en plusieurs sons plus petits (de 4 secondes)

    Parameters
    ----------
    data : ndarray
        -- le son que l'on souhaite diviser
    sr : int
        -- la fréquence d'échantillonage utilisé
    audio_duration : float
        -- durée souhaitée des sons en sorties
    Returns
    -------
    divide_sound: list
        -- liste des petits sons 
    '''
    nombre_points = len(data)
    points_esperes = sample_freq * audio_duration
    if nombre_points >= points_esperes:
        audio1 = data[:points_esperes]
        return [audio1] + diviser_son(data[points_esperes:])
    else:
        return []


def feu_to_dataset(): #on transfère les sons contenant du feu dans le dataset (en les divisant en sons plus petits)
    for count, filename in enumerate(os.listdir("Feu")):
        print(count)
        data,samplerate=sf.read(f"Feu/{filename}")
        (i,j)=silence(data[:,0])
        signaux=diviser_son(data[i:j+1])
        for k in range(len(signaux)):
            wavio.write(f"Dataset/{filename[:-4]}_{k}.wav", signaux[k], samplerate, sampwidth=4)

def vider_poubelle(poubelle): #si jamais on souhaite récupérer des fichiers des poubelles
    folder="audio/"+poubelle
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        fichier=filename.split('-',3)
        src =f"{folder}/{filename}" 
        dst =f"{folder2}{int(fichier[1])+1}/{filename}"
        os.rename(src,dst)

def autre_to_dataset(liste): 
    '''
    On transfère les fichiers ne contenant pas de feu dans le dataset

    Parameters
    ----------
    liste : list
        -- le nombre de sons à prendre par classe

    Returns
    -------
    None
    '''
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

choix2=[400,175,559,408,400,400,16,400,400,400] #cela correspond au nombres de fichiers audio par classe que l'on transfère dans le dataset

def compter_dataset(folder): #pour connaître le nombre de sons de feu et de sons de pas feu
    feu=0
    autre=0
    for count, filename in enumerate(os.listdir(folder)):
        if filename[:3]=="Feu":
            feu+=1
        elif filename[:3]=="Aut":
            autre+=1
    print('Il y a '+str(feu)+" feu")
    print("Il y a "+str(autre)+" autre")


def dataset_to_autre(folder): #Pour enlever les fichiers ne contenant pas de feu du dataset (et remmetre les sons dans leurs dossiers respectifs)
    print("vide le dataset")
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        print("boucle")
        if filename[:3]=="Aut":
            print("if")
            fichier=filename.split('-',3)
            src =f"{folder}/{filename}"  
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
    '''
    On vérifie que le dataset correpond à nos attentes()

    Parameters
    ----------
    folder : string
        -- chemin du dosiier que l'on souhaite verifier
    freq_echant : int
        -- frequence souhaitée
    echant_temps : float
        -- temps souahité

    Returns
    -------
    Print les problèmes du dataset ainsi que la répartition feu/pas feu
    '''
    print("Début de check_dataset")
    sil,freq,tem,feu,autre,nom,ext=0,0,0,0,0,0,0
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
    if ext!=0:
        print("Il y a "+str(ext)+" fichiers qui ne finissent pas en .wav")
    if nom!=0:
        print("Il y a "+str(nom)+" fichiers qui ne sont pas renommés correctement")
    if tem!=0:
        print("Il y a "+str(tem)+" sons qui n'ont pas la bonne durée")
    if freq!=0:
        print("Il y a "+str(freq)+" sons qui ne sont pas à la bonne fréquence d'échantillonnage")
    if sil!=0:
        print("Il y a "+str(sil)+" sons contenant des silences au début ou à la fin")
    print("La proportion feu/pas feu est de "+str(feu)+"/"+str(autre))
    print("fin")


def training_val_test(Feu,Autre): #On divise le dataset en trois (Training/Validation/Test)
    #on prend en entrée le nombre de sons de feu et de pas feu (donc des entiers)
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
        


def vider_training_val_test(): #on vide les sous dossiers du dataset 
    folder="Dataset"
    list=["Training","Validation","Test"]
    for i in range(3):
        for count, filename in enumerate(os.listdir(f"{folder}/{list[i]}")):
            src =f"{folder}/{list[i]}/{filename}"  
            dst =f"{folder}/{filename}"
            os.rename(src,dst)


def autre_to_dataset2(liste): #on transfère les fichiers ne contenant pas de feu dans le dataset (on prend en compte la répartition training/validation/test)
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
    #on prend en compte la répartition training/validation/test
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
        signaux=diviser_son(data[i:j+1])
        for k in range(len(signaux)):
            wavio.write(f"{folder}/{filename[:-4]}_{k}.wav", signaux[k], samplerate, sampwidth=4)

def delete_autre(folder): #On supprime les fichiers proveant de classes qui ne nous intéresse plus
    print("vide le dataset")
    folder2="audio/fold"
    for count, filename in enumerate(os.listdir(folder)):
        if filename[:3]=="Aut":
            fichier=filename.split('-',3)
            print(fichier)
            if int(fichier[1]) in [0,4,5,7,8,9]:
                src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
                dst =f"{folder2}{int(fichier[1])+1}/{filename}"
                os.rename(src,dst)

def mp3_to_wav(folder): #transforme les sons .mp3 en .wav
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}" 
        dst =f"{folder}/{filename[:-3]}wav"
        os.rename(src,dst)


def melanges(folder,folder2,folder3):
    '''
    On mélange des sons de feu avec d'autre sons pour observer les résultats obtenues pour des feux avec un bruit de fond (bruit d'oiseaux par exemple)

    Parameters
    ----------
    folder : string
        -- chemin du dossier contenant les feux
    folder2 : string
        -- chemin du dossier contenant les bruits de fond à ajouter
    folder3 : string
        -- chemin du dossier dans lequel on stock le résultat

    Returns
    -------
    None
    '''
    liste=[]
    c=0
    for count, filename in enumerate(os.listdir(folder2)):
        liste.append(f"{folder2}/{filename}")
    for count, filename in enumerate(os.listdir(folder)):
        data,samplerate=sf.read(f"{folder}/{filename}")
        data2,samplerate2=sf.read(f"{liste[c]}")
        c+=1
        print(c)
        signal=data[:176400]
        for k in range(len(data2)):
            signal[k]=(data[k]+0.1*data2[k][0])
            #signal[k][1]=data[k][1]+data2[k][1] #si le son est en stéréo
        wavio.write(f"{folder3}/Feu{filename[:-4]}{count}.wav", signal, samplerate, sampwidth=4)