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

# for i in range(len(wines)):
#   print( wines[i]["title"])

client = MongoClient("mongodb://localhost:27017")
dbase = client["wineDB"]
myPopularCollection = dbase["popularWineCollection"]

all_imgs = os.listdir(IMG_PATH)



def send_data_db():
    # img_loc =os.path.join(IMG_PATH, all_imgs[1])
    # image = open(img_loc, 'rb')
    # print(image)
    # image_read = image.read()
    # image_64_encode = base64.b64encode(image_read)
    # print(image_64_encode)

    for idx in range(len(all_imgs)):
        img_loc =os.path.join(IMG_PATH, all_imgs[idx])
        image = open(img_loc, 'rb')
        image_read = image.read()
        image_encode = base64.b64encode(image_read).decode("ascii")
        # image_encode = base64.b64encode(image_read)
        # im = Image.open(img_loc).convert("RGB")
        # im= im.resize((IMG_WIDTH,IMG_HEIGHT))
        # image_bytes = io.BytesIO()
        # im.save(image_bytes, format='JPEG')
        # wines[idx]["points"] = np.random.randint(1,6)
        wines[idx]["image"] = image_encode
        # wines[idx]["image"] = image_bytes.getvalue()
        final_wines.append(wines[idx])
    # myPopularCollection.insert_many(final_wines)

send_data_db()


# def resize_image():

#      for idx in range(len(all_imgs)):
#         img_loc =os.path.join(IMG_PATH, all_imgs[idx])
#         new_loc = os.path.join('./Final_data_for_train/', all_imgs[idx])
#         # image = open(img_loc, 'rb')
#         # image_read = image.read()
#         # image_encode = base64.b64encode(image_read).decode("ascii")
#         im = Image.open(img_loc).convert("RGB")
#         im= im.resize((IMG_WIDTH,IMG_HEIGHT),Image.ANTIALIAS)
#         im.save(new_loc)
#         # image_result = open(new_loc, 'wb') # create a writable image and write the decoding result
#         # image_result.write(im)

#         # image_bytes = io.BytesIO()
#         # im.save(image_bytes, format='JPEG')
#         # wines[idx]["points"] = np.random.randint(1,6)
#         # wines[idx]["image"] = image_encode
#         # wines[idx]["image"] = image_bytes.getvalue()
#         final_wines.append(wines[idx])



# resize_image()



# # send_data_db()
# print("succesfully")


# def get_image():

#     img_loc =os.path.join(IMG_PATH, all_imgs[1])
#     image = open(img_loc, 'rb',)

#     new_loc = os.path.join('./Final_data_for_train/', all_imgs[1])
    
#     print(image)
#     image_read = image.read()
#     image_read= image_read.resize((IMG_WIDTH,IMG_HEIGHT))
#     image_64_encode = base64.b64encode(image_read).decode('ascii')
#     print(image_64_encode)
#     image_64_decode = base64.b64decode(image_64_encode) 
#     image_result = open('deere_decode.jpg', 'wb') # create a writable image and write the decoding result
  
#     image_result.write(image_64_decode)
#     # plt.imshow(image_result)
#     # plt.show()

#     # encoded_string = base64.b64encode(im.read()).decode("ascii")
#     # print(image_read)

# # # print(final_wines)
# # get_image()


# def another_test():
#     for idx in range(5):
#         img_loc =os.path.join(IMG_PATH, all_imgs[idx])
#         im = Image.open(img_loc).convert("RGB")
#         encoded_string = base64.b64encode(im)
#         print(encoded_string)
#         im= im.resize((IMG_WIDTH,IMG_HEIGHT))
#         image_bytes = io.BytesIO()
#         # im.save(image_bytes, format='JPEG')
#         # wines[idx]["points"] = np.random.randint(1,6)
#         # wines[idx]["image"] = image_bytes.getvalue()
#         # final_wines.append(wines[idx])


# # another_test()



# # for idx in range(len(all_imgs)):
# #     img_loc =os.path.join(IMG_PATH, all_imgs[idx])
# #     # print(img_loc)
# #     with open(img_loc, 'rb') as image_file:
# #         encoded_string = base64.b64encode(image_file.read()).decode("ascii")
# #         products["rate"] =1
# #         products["image"] = encoded_string
# #  
# #        wines.append(products)

# # client = MongoClient("mongodb://localhost:27017")
# # mydb = client["imagepyDB"]
# # mycollections = mydb["wine_image"]

# # for idx in range(len(all_imgs)):
# #     img_loc =os.path.join(IMG_PATH, all_imgs[idx])
# #     im = Image.open(img_loc)
# #     image_bytes = io.BytesIO()
# #     im.save(image_bytes, format='JPEG')
# #     image = {
# #         "image":image_bytes.getvalue()
# #     }
# #     wines.append(image)

# # mycollections.insert_many(wines)








# # img_loc = os.path.join(self.main_dir, all_imgs[idx])
# # image = Image.open(img_loc).convert("RGB")
# # image = image.resize((IMG_WIDTH,IMG_HEIGHT))