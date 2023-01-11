import cv2 as cv
import os
from tqdm import tqdm
import numpy as np

#resize l'image en place, les dimensions sont choisies au vu du format du spectrogramme pour ne garder que 
#la partie du spectrogramme correspondant aux fréquences qui nous intéressent (critère de Shannon)
def resize_en_place(image):
    resized=image[172:593,105:1035]
    scale_percent = 50 # percent of original size
    width = int(resized.shape[1] * scale_percent / 100)
    height = int(resized.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv.resize(resized,dim,interpolation=cv.INTER_AREA)
    
# récupère le path pour modifier l'image et la réupload au même endroit
def resize_path(image_path):
    image= cv.imread(image_path)
    resized = resize_en_place(image)
    cv.imwrite(image_path,resized)

# parcours le dossier et applique la fonction resize_path à chaque image
def transform_db(INPUT_DB_PATH):
    directory = os.fsencode(INPUT_DB_PATH)
    listd_dir=os.listdir(directory)
    for spec_path in tqdm(listd_dir):
        spec=os.fsdecode(spec_path)
        resize_path(INPUT_DB_PATH+"/"+spec)

if __name__=='__main__':
    INPUT_DB_PATH="Dataset_spec/Test"
    transform_db(INPUT_DB_PATH)
