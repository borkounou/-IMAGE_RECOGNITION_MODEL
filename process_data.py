# Contains dataset class that can be used for loading data

__all__ = ["FolderDataset"]

import torch 
from PIL import Image
import os 
from tqdm import tqdm
from torch.utils.data import Dataset
from config import IMG_HEIGHT, IMG_WIDTH
import config 

main_dir = './wine_data/train_and_valid/X_valid/'
new_dir ="./Final_data_for_train/"

u =130
for dirname in os.listdir(main_dir):
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(u) + ".jpg")
            print(filename)
            u =u+1



