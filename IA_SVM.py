import librosa
import numpy as np
import pandas as pd
from pandas import DataFrame

def predic_svm(path_file,model):

    '''
    prediction du modèle svm sur un fichier audio

    Parameters
    ----------
    path_file : string
        -- path of the file
    model : model (SVM object) 
        -- model choisi
    
    Returns
    -------
    prediction : int
        -- prediction du model (0 si pas de feu, 1 si feu)
    '''


    WAV_PATH = path_file

    def calcul_features_saving_csv(path_file, path_saved_features): #paths have to be string and without 'csv'
        '''
        creation des features et sauvegarde dans un csv

        Parameters
        ----------
        path_file : string
            -- path of the file
        path_saved_features : string
            -- path of the csv file where the features will be saved
        
        Returns
        -------
        None. (saves a csv file)
        
        '''
        zcr = [] #zero crossing rate
        mean_spectral_centroids = [] # moyenne du Spectral centroid
        rolloff_point = [] #calcul du spectral rolloff point
        mfcc = [ [] for _ in range(20) ] #liste qui contiendra les listes de mfcc (mfcc1, mfcc2,...)

    

        audio = librosa.load(path_file)[0]

        # Calcul du ZCR
        zcr0 = librosa.zero_crossings(audio)
        zcr.append(sum(zcr0))

            # Calcul de la moyenne du Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(audio)[0]
        mean_spectral_centroids.append(np.mean(spectral_centroids))

            # Calcul du spectral rolloff point

        rolloff = librosa.feature.spectral_rolloff(audio)
        rolloff_point.append(np.mean(rolloff))

            # Calcul des moyennes des MFCC

        mfcc0 = librosa.feature.mfcc(audio) #liste de 20 liste (mfcc1, mfcc2, ...)

        for i in range(len(mfcc0)):
            mfcc[i].append(np.mean(mfcc0[i]))

        #checking if there is no error of list size
        n = 1
        filenames = [path_file]
        assert len(zcr) == n
        assert len(rolloff_point) == n
        assert len(mean_spectral_centroids) == n

        for i in range(len(mfcc)):
            assert len(mfcc[i]) == n

        #preparing dictionnary to create the dataframe
        dico1 = {'Filesnames' : filenames,
        'zcr': zcr,
        'spectral_c':mean_spectral_centroids,
        'rolloff': rolloff_point
        }
        dico2 = {
            'mfcc' + str(i): mfcc[i-1] for i in range(1,21)
        }

        data_features_dico = dico1 | dico2

        data_features_df = DataFrame(data_features_dico, columns= ['Filesnames', 'zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
                    'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                    'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                    'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'])

        export_csv_features = data_features_df.to_csv (path_saved_features + '.csv' , index = None, header=True, encoding='utf-8')

        #labelisation of the dataset


        #adding the column of the labels in the dataframe 

        #saving csv with labels

        
        
    #calcul des features et sauvegarde dans un csv
    calcul_features_saving_csv(WAV_PATH,"features_wav")

    #chargement du csv et choix des features
    features=['zcr','rolloff','mfcc2','mfcc12']
    X_train_full = pd.read_csv('features_wav.csv').iloc[:,1:25]
    X_train = X_train_full.loc[:, features]

    #prediction du modèle
    y_pred = model.predict(X_train)[0]
    if y_pred == 1:
        return [1]
    else:
        return [0]
