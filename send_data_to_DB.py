from data_for_wine import wines
from PIL import Image
from pymongo import MongoClient
import io 
import base64
from config import IMG_HEIGHT, IMG_WIDTH, IMG_PATH
import os 
import numpy as np 
import matplotlib.pyplot as plt
import base64



print(len(wines))
final_wines = []
final_dict = {}

client = MongoClient("mongodb://localhost:27017")
dbase = client["wineDB"]
myPopularCollection = dbase.popularWineCollection#["popularWineCollection"]

all_imgs = os.listdir(IMG_PATH)
# wine = myPopularCollection.find_one()

# my_query = {"price":wine["price"]}
# new_value = {"$set":{"price":1000}}
# myPopularCollection.update_one(my_query, new_value)
# for x in myPopularCollection.find():
#     print(x['price'])



def send_data_db():
    for idx, wine in enumerate(myPopularCollection.find()):
        img_loc =os.path.join(IMG_PATH, all_imgs[idx])
        image = open(img_loc, 'rb')
        image_read = image.read()
        image_encode = base64.b64encode(image_read).decode("ascii")
        my_query = {"image":wine["image"]}
        new_value = {"$set":{"image":image_encode}}
        # my_query = {'name':all_imgs[idx]}
        # wine["name"] = all_imgs[idx]
        # myPopularCollection.update_one({"$set":{"name":all_imgs[idx]}})
        # myPopularCollection.update_one({"name":""},{"$set":{"name":str(all_imgs[idx])}} )
        myPopularCollection.update_one(my_query, new_value)
    print("successfull update")



    # for idx in range(len(all_imgs)):
    #     img_loc =os.path.join(IMG_PATH, all_imgs[idx])
    #     image = open(img_loc, 'rb')
    #     image_read = image.read()
    #     image_encode = base64.b64encode(image_read).decode("ascii")




        # wines[idx]["points"] = np.random.randint(1,6)
    #     wines[idx]["image"] = image_encode
    #     final_wines.append(wines[idx])
    # myPopularCollection.insert_many(final_wines)
    # print("Data uploaded successfully to the data")


send_data_db()