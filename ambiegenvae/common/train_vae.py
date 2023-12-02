import argparse
import torch
import os
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# The ToTensor1D class is a custom transform that converts a 1D vector to a PyTorch tensor
class ToTensor1D(object):
    def __call__(self, sample):
        # Convert the 1D vector to a PyTorch tensor
        return torch.FloatTensor(sample)

class Normalize1D(object):
    def __init__(self, mean: float, std: float):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, sample):

        return (sample - self.mean) / self.std


class Denormalize1D(object):
    def __init__(self, mean: float, std: float):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, sample):
        # Apply Z-score normalization
        return (sample * self.std) + self.mean


class VecDataSet(Dataset):
    """
     # The `VecDataSet` class is a custom dataset class in Python that takes in a numpy array of data and
    # applies optional data transformations before returning the data and a label.
    """

    def __init__(self, data, data_transforms=None):
        self.data = data
        self.data_transforms = data_transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        out_data = np.array(self.data[idx])

        if self.data_transforms:
            out_data = self.data_transforms(out_data)

        return out_data, 0
    

class VecVAESimple(nn.Module):
    """
    The code defines a Variational Autoencoder (VAE) model in PyTorch for encoding and decoding 1D
    vectors.

    :param input_space: The input space refers to the dimensionality of the input vectors. It represents
    the number of features or elements in each input vector
    :param latent_space: The `latent_space` parameter represents the dimensionality of the latent space.
    It determines the size of the bottleneck layer in the Variational Autoencoder (VAE). The latent
    space is a lower-dimensional representation of the input data that captures the most important
    features
    :param dataset: The `dataset` parameter is an optional argument that allows you to pass a custom
    dataset to the VecVAE class. If no dataset is provided, it defaults to `VecDataSet`, which is
    assumed to be a dataset of 1-dimensional vectors
    """

    def __init__(self, input_space_size, latent_space_size):
        super(VecVAESimple, self).__init__()

        self.input_space = input_space_size
        self.latent_space = latent_space_size
        self.hidden_size = self.latent_space * 2

        self.fc1 = nn.Linear(self.input_space, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.latent_space)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_space)
        self.fc3 = nn.Linear(self.latent_space, self.hidden_size)
        #self.fc31 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_space)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        #h2 = F.relu(self.fc11(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #h4 = F.relu(self.fc31(h3))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# https://github.com/pytorch/examples/blob/master/vae/main.py
# The VecVAE class is a variational autoencoder that takes in 1D vectors as input and maps them to a
# lower-dimensional latent space.
class VecVAE(nn.Module):
    """
    The code defines a Variational Autoencoder (VAE) model in PyTorch for encoding and decoding 1D
    vectors.

    :param input_space: The input space refers to the dimensionality of the input vectors. It represents
    the number of features or elements in each input vector
    :param latent_space: The `latent_space` parameter represents the dimensionality of the latent space.
    It determines the size of the bottleneck layer in the Variational Autoencoder (VAE). The latent
    space is a lower-dimensional representation of the input data that captures the most important
    features
    :param dataset: The `dataset` parameter is an optional argument that allows you to pass a custom
    dataset to the VecVAE class. If no dataset is provided, it defaults to `VecDataSet`, which is
    assumed to be a dataset of 1-dimensional vectors
    """

    def __init__(self, input_space_size, latent_space_size):
        super(VecVAE, self).__init__()

        self.input_space = input_space_size
        self.latent_space = latent_space_size
        self.hidden_size = self.latent_space * 2

        self.fc1 = nn.Linear(self.input_space, self.hidden_size)
        self.fc11 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.latent_space)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_space)
        self.fc3 = nn.Linear(self.latent_space, self.hidden_size)
        self.fc31 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_space)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc11(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        return torch.tanh(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DeepVecVAE(nn.Module):
    def __init__(self, input_space_size, latent_space_size):
        super(DeepVecVAE, self).__init__()

        self.input_space = input_space_size
        self.latent_space = latent_space_size
        self.hidden_size = self.latent_space * 2

        # Encoder
        self.fc1 = nn.Linear(self.input_space, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, self.latent_space)
        self.fc32 = nn.Linear(64, self.latent_space)

        # Decoder
        self.fc4 = nn.Linear(self.latent_space, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, self.input_space)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.tanh(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_functionA(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    return BCE + KLD
def loss_functionB(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    return BCE +  0.1* KLD

def load_model(epoch, model, path='.//'):

    # creating the file name indexed by the epoch value
    filename = path + '/neural_network_{}.pt'.format(epoch)

    # loading the parameters of the saved model
    model.load_state_dict(torch.load(filename))

    return model


def save_model(epoch, model, path='results'):

    if not (os.path.exists(path)):
        os.makedirs(path)

    # creating the file name indexed by the epoch value
    filename = path + '/neural_network_{}.pt'.format(epoch)

    # saving the model parameters
    torch.save(model.state_dict(), filename)

    return model
def train_step(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_function,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        #print("loss", loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_step(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_function
):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    # print("====> Test set loss: {:.4f}".format(test_loss))

    return test_loss




def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function,
    optimizer: torch.optim.Optimizer,
    epochs: int,
):
    """
    Train a PyTorch model for a number of epochs.
    """

    results = {"train_loss": [], "test_loss": []}

    #loss_fn = loss_function

    # Put model in training mode
    model.train()
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            train_loader=train_loader,
            #loss_function=loss_fn,
            optimizer=optimizer,
        )
        test_loss = test_step(model=model, test_loader=test_loader) #, loss_function=loss_fn)

        # Checkpoint
        #if epoch % checkpoint_freq == 0:
        #    save_model(epoch, model, result_folder)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # 6. Return the filled results at the end of the epochs
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAE Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=2023, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=5,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--data", type=str, default="vae_dataset.npy", metavar="N", help="data"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    nDim = 17  # Dimension of the input vector for VAE
    nLat = 17  # Dimension of the latent space for VAE

    # Load data set
    archive = np.load(args.data)
    data_train = archive[: int(0.8 * len(archive))]
    data_test = archive[int(0.8 * len(archive)) :]
    mean = np.mean(archive, axis=0)  # mean for each feature
    std = np.std(archive, axis=0)

    # Normalize data
    transforms = transforms.Compose([ToTensor1D(), Normalize1D(mean, std)]) # 
    dataset_train = VecDataSet(data_train, data_transforms=transforms)
    dataset_test = VecDataSet(data_test, data_transforms=transforms)

    train_load = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_load = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    out, label_custom = next(iter(train_load))
    model = DeepVecVAE(nDim, nLat).to(device)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = loss_functionA
    results = train(
        model=model,
        train_loader=train_load,
        test_loader=train_load,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=args.epochs,
    )


