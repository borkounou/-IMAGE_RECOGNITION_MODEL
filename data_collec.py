import base64
from PIL import Image
from config import IMG_HEIGHT, IMG_WIDTH, IMG_PATH
import numpy as np 
from pymongo import MongoClient

import io
import os 

all_imgs = os.listdir(IMG_PATH)

# print(all_imgs)
wines = []
products = {};
print(len(all_imgs))
# def get_data(name, image):
#     dict_1 = {
#         "wines":[
#             {
#                 "name": name,
#                 "image":image,
#             }
#         ]
#     }

#     return dict_1


# for idx in range(len(all_imgs)):
#     img_loc =os.path.join(IMG_PATH, all_imgs[idx])
#     # print(img_loc)
#     with open(img_loc, 'rb') as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode("ascii")
#         products["rate"] =1
#         products["image"] = encoded_string
#  
#        wines.append(products)

client = MongoClient("mongodb://localhost:27017")
mydb = client["imagepyDB"]
mycollections = mydb["wine_image"]

for idx in range(len(all_imgs)):
    img_loc =os.path.join(IMG_PATH, all_imgs[idx])
    im = Image.open(img_loc)
    image_bytes = io.BytesIO()
    im.save(image_bytes, format='JPEG')
    image = {
        "image":image_bytes.getvalue()
    }
    wines.append(image)

mycollections.insert_many(wines)








# img_loc = os.path.join(self.main_dir, all_imgs[idx])
# image = Image.open(img_loc).convert("RGB")
# image = image.resize((IMG_WIDTH,IMG_HEIGHT))