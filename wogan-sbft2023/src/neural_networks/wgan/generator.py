import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorNetwork(nn.Module):
    """
    Define the neural network model for the WGAN generator.
    """

    def __init__(self, input_shape, output_shape, neurons):
        super(GeneratorNetwork, self).__init__()

        # The dimension of the input vector.
        self.input_shape = input_shape
        # The dimension of the output vector.
        self.output_shape = output_shape
        # Number of neurons per layer.
        self.neurons = neurons

        # We use three fully connected layers with self.neurons many neurons.
        self.glayer1 = nn.Linear(self.input_shape, self.neurons)
        self.glayer2 = nn.Linear(self.neurons, self.neurons)
        self.glayer3 = nn.Linear(self.neurons, self.output_shape)
        self.bn1 = nn.BatchNorm1d(self.neurons)
        self.bn2 = nn.BatchNorm1d(self.neurons)
        self.bn3 = nn.BatchNorm1d(self.output_shape)

    def forward(self, x):
        x = F.relu(self.bn1(self.glayer1(x)))
        x = F.relu(self.bn2(self.glayer2(x)))
        x = torch.tanh(self.bn3(self.glayer3(x)))  # Squash the values to [-1, 1].

        return x
