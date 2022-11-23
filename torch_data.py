# Contains dataset class that can be used for loading data

__all__ = ["FolderDataset"]

import torch 
from PIL import Image
import os 
from tqdm import tqdm
from torch.utils.data import Dataset
from config import IMG_HEIGHT, IMG_WIDTH
import config 

class FolderDataset(Dataset):

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    
    def __len__(self):
        return len(self.all_imgs)
    

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((IMG_WIDTH,IMG_HEIGHT))
        if self.transform is not None:
            tensor_image = self.transform(image)
        
        return tensor_image, tensor_image
