#contains training and validation steps. It also contains code to create embeddings.

__all__ = ["train_step", "val_step", "create_embedding"]

import torch
import torch.nn as nn 

def train_step(encoder,decoder, train_loader, loss_fn, optimizer, device):

    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        optimizer.zero_grad()
        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)
        loss = loss_fn(dec_output, target_img)
        loss.backward()
        optimizer.step()

    return loss.item()



def val_step(encoder, decoder, val_loader, loss_fn, device):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)
    return loss.item()





def create_embedding(encoder, full_loader, embedding_dim, device):

    encoder.eval()
    embedding = torch.randn(embedding_dim)
    print(embedding.shape)

    with torch.no_grad():

        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            # print(enc_output.shape)
            embedding = torch.cat((embedding, enc_output), 0)
            # print(embedding.shape)
    return embedding
