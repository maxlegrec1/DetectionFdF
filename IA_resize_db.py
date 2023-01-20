import cv2 as cv
import os
from tqdm import tqdm



def resize_en_place(image):
    '''
    Crop l'image pour ne garder que la partie du spectrogramme correspondant aux fréquences qui nous intéressent (critère de Shannon)

    Parameters
    ----------
    image : numpy array
        -- image du spectrogramme. 

    Returns
    -------
    numpy array
        -- image du spectrogramme redimensionnée.
    
    '''
    resized=image[190:593,110:1035]
    scale_percent = 50 # percent of original size
    width = int(resized.shape[1] * scale_percent / 100)
    height = int(resized.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv.resize(resized,dim,interpolation=cv.INTER_AREA)
    
def resize_path(image_path):
    '''
    récupère le path pour modifier l'image et la réupload au même endroit

    Parameters
    ----------
    image_path : string
        -- chemin de l'image à redimensionner.

    Returns
    -------
    None. (redimensionne l'image et la réupload au même endroit)
    
    '''
    image= cv.imread(image_path)
    resized = resize_en_place(image)
    cv.imwrite(image_path,resized)

def transform_db(INPUT_DB_PATH):
    '''
    parcoure le dossier et applique la fonction resize_path à chaque image

    Parameters
    ----------
    INPUT_DB_PATH : string
        -- chemin du dossier contenant les images à redimensionner.
    
    Returns
    -------
    None. (redimensionne les images et les réupload au même endroit)

    '''
    directory = os.fsencode(INPUT_DB_PATH)
    listd_dir=os.listdir(directory)
    for spec_path in tqdm(listd_dir):
        spec=os.fsdecode(spec_path)
        resize_path(INPUT_DB_PATH+"/"+spec)

if __name__=='__main__':
    INPUT_DB_PATH="Dataset_spec/Test"
    transform_db(INPUT_DB_PATH)
