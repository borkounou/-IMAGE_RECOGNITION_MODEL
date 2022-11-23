from pymongo import MongoClient
from flask import Flask, jsonify, request, json,json
from json import *
from bson import json_util

client = MongoClient("mongodb://localhost:27017")
dbase = client["wineDB"]
myPopularCollection = dbase["popularWineCollection"]



s = myPopularCollection.find_one({"name":"2.jpg"},{})


# data = json.loads(json_util.dumps(s))
#     print(wine["name"][0])
# document = myPopularCollection.find_one({"name":"2.jpg"})
data = json.loads(json_util.dumps(s))

data = json.dumps(data)
print(data)
# dat = json.dumps(data)

# da = json.dumps(data)
# print(da)
# print(type(da))
# print(dat["image"])

# print(data["image"])
# # print(dat)
# r = jsonify({"image": data["image"]})
# # print(type(dat))


# for  k, v in data.items():
#     print(v)
