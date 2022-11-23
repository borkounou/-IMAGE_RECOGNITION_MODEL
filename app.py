
from flask import Flask, request, json
import torch_model
import config
import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
from sklearn.decomposition import PCA

from pymongo import MongoClient

imageFile = ""

app = Flask(__name__)

print("App started")

#Geting data from the database 

client = MongoClient("mongodb://localhost:27017")
myDB = client.wineDB
myPopularCollection = myDB.popularWineCollection

# Device : GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model before we start the server
encoder = torch_model.ConvEncoder()
# Load the state dict of encoder
encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
# Loads the embedding
embedding = np.load(config.EMBEDDING_PATH)

print("Loaded model and embeddings")


def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.
    Args:
    image_tenosr: PIL read image_tensor whose similar images are needed.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list

# For the home route and health check
@app.route("/")
def index():
    return "App is Up"



@app.route("/simimages", methods=["GET", "POST"])
def simimages():
    global imageFile

    if(request.method == "POST"):
        image = request.files["image"]
        # print("Hi")
        image = Image.open(image)
        image_tensor = T.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        indices_list = compute_similar_images(image_tensor, num_images=5, embedding=embedding, device=device)
        print(indices_list)
    # Need to display the images
        return (
            json.dumps({"indices_list": indices_list}),
            200,
            {"ContentType": "application/json"},
        )


if __name__ == "__main__":
    app.run(debug=False, port=5000)