__all__ = [
    "load_image_tensor",
    "compute_similar_images",
    "compute_similar_features",
    "plot_similar_images",
]

import config
import torch
import numpy as np
import torch_model
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_image_tensor(image_path, device):
    img = Image.open(image_path)
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    image_tensor = T.ToTensor()(img)
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def compute_similar_images(image_path, num_images, embedding,encoder, device):

    image_tensor = load_image_tensor(image_path, device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    print(indices_list)
    return indices_list


def plot_similar_images(indices_list):
    indices = indices_list[0]
    for index in indices:
        print("index is: ", index)
        if index==0:
            pass 
        else:

            img_name = str(index-1)+".jpg"
            print(img_name)
            img_path = os.path.join(config.DATA_PATH+ img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
            plt.imshow(img)
            plt.show()
            img.save(f"./outputs/recommended_{index-1}.jpg")


if __name__ == "__main__":
    # Loads the model 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder = torch_model.ConvEncoder()
    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)
    # Loads the embedding
    embedding = np.load(config.EMBEDDING_PATH)
    
    indices_list = compute_similar_images(config.TEST_IMAGE_PATH, config.NUM_IMAGES, embedding, encoder, device)

    plot_similar_images(indices_list)


