
import sys
import argparse
import torch
import math
import os
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def plot_latent_space(model: torch.nn.Module, test_data: torch.utils.data.DataLoader, epoch:int, path:str ='results'):

    if not (os.path.exists(path)):
        os.makedirs(path)

    model.eval()

# Get the latent space data
    latent_space_data = []
    i  = 0
    with torch.no_grad():
        for batch in test_data:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Encode the data
            #i += 1
            #if i > 5:
            #    break

            mu, _ = model.encode(inputs)
            #print(mu)
            # Append the latent space data to the list
            latent_space_data.append(mu.cpu())

    latent_space_data = torch.cat(latent_space_data, dim=0).numpy()

    # Visualize the latent space

    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_space_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], cmap='jet')
    plt.xlim(-40, 40)  # Replace 'xmin' and 'xmax' with your desired range

    # Set the maximum range for the y-axis
    plt.ylim(-40, 40)

    #plt.colorbar()
    #plt.show()
    save_path = os.path.join(path, str(epoch) + ".png")
    plt.savefig(save_path)

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']


    # Get the accuracy values of the results dictionary (training and test)

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))


    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

def calculate_latent_space(model, test_load):

    # Get the latent space data
    latent_space_mu_data = []
    latent_space_std_data = []
    i  = 0
    with torch.no_grad():
        for batch in test_load:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Encode the data
            #i += 1
            #if i > 8:
            #    break

            mu, logvar = model.encode(inputs)#.cpu().numpy()
            std = torch.exp(0.5*logvar)
            #print(std)
            # Append the latent space data to the list
            latent_space_mu_data.append(mu.cpu())
            latent_space_std_data.append(std.cpu())

    latent_space_mu_data = torch.cat(latent_space_mu_data, dim=0).numpy()
    latent_space_std_data = torch.cat(latent_space_std_data, dim=0).numpy()
    # calculate the mean of the latent space
    mean_l = np.mean(latent_space_mu_data, axis=0)
    #print("Mean", mean_l)
    #print(np.mean(mean_l))
    # calculate the standard deviation of the latent space
    #std_l = np.mean(latent_space_std_data, axis=0)
    std_l = np.std(latent_space_mu_data, axis=0)
    #print("Std", std_l)
    return np.mean(mean_l), np.mean(std_l)