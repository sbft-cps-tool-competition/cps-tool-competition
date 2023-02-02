import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Define the neural network model for the WGAN critic.
    """

    def __init__(self, input_shape, neurons):
        super(CriticNetwork, self).__init__()

        # The dimension of the input vector.
        self.input_shape = input_shape
        # Number of neurons per layer.
        self.neurons = neurons

        # We use three fully connected layers with self.neurons many neurons.
        self.clayer1 = nn.Linear(self.input_shape, self.neurons)
        self.clayer2 = nn.Linear(self.neurons, self.neurons)
        self.clayer3 = nn.Linear(self.neurons, 1)

    def forward(self, x):
        x = F.leaky_relu(self.clayer1(x),
                         negative_slope=0.1)  # LeakyReLU recommended in the literature for GAN discriminators.
        x = F.leaky_relu(self.clayer2(x), negative_slope=0.1)
        x = self.clayer3(x)  # No activation for the critic.

        return x
