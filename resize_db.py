import cv2 as cv
import os
from tqdm import tqdm
def resize_en_place(image_path):
    image= cv.imread(image_path)
    resized=image[14:593,105:1035]
    scale_percent = 50 # percent of original size
    width = int(resized.shape[1] * scale_percent / 100)
    height = int(resized.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized=cv.resize(resized,dim,interpolation=cv.INTER_AREA)
    cv.imwrite(image_path,resized)


def transform_db(INPUT_DB_PATH):
    directory = os.fsencode(INPUT_DB_PATH)
    listd_dir=os.listdir(directory)
    for spec_path in tqdm(listd_dir):
        spec=os.fsdecode(spec_path)
        resize_en_place(INPUT_DB_PATH+"/"+spec)

if __name__=='__main__':
    INPUT_DB_PATH="Dataset_spec/Training"
    transform_db(INPUT_DB_PATH)