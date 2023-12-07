import os

import numpy as np
import torch

from src.neural_networks.wgan.analyzer import Analyzer_NN_weighted_new
from src.neural_networks.wgan.critic import CriticNetwork
from src.neural_networks.wgan.generator import GeneratorNetwork


class Model:
    """
  Base class for all models.
  """

    def __init__(self, sut, validator, device, logger=None):
        self.sut = sut
        self.device = device
        self.validator = validator
        self.logger = logger
        self.log = lambda t: logger.info(t) if logger is not None else None

        self.saved_parameters = ["algorithm_version", "train_settings", "train_settings_init"]

        # Settings for training. These are set externally.
        self.algorithm_version = None
        self.train_settings_init = None
        self.train_settings = None
        self.random_init = None
        self.N_tests = None

    def initialize(self):
        pass

    @property
    def parameters(self):
        return {k: getattr(self, k) for k in self.saved_parameters if hasattr(self, k)}

    def train_with_batch(self, dataX, dataY, train_settings, log=False):
        raise NotImplementedError()

    def generate_test(self, N=1):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def validity(self, tests):
        """
    Validate the given test using the true validator.
    Args:
      tests (np.ndarray): Array of shape (N, self.sut.ndimensions).
    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

        if len(tests.shape) != 2 or tests.shape[1] != self.sut.ndimensions:
            raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

        if self.validator is None:
            result = np.ones(shape=(tests.shape[0], 1))
        else:
            result = self.validator.validity(tests)

        return result


class WGAN(Model):
    """
  Implements the WGAN model.
  """

    def __init__(self, sut, validator, device, logger=None):
        super().__init__(sut, validator, device, logger)

        # These parameters are set externally.
        self.saved_parameters += ["noise_dim", "gan_neurons", "gp_coefficient", "gan_learning_rate",
                                  "analyzer_learning_rate", "analyzer_neurons"]
        # Input dimension for the noise inputted to the generator.
        self.noise_dim = None
        # Number of neurons per layer in the neural networks.
        self.gan_neurons = None
        # The coefficient for the loss gradient penalty term.
        self.gp_coefficient = None
        # Learning rate for WGAN optimizers.
        self.gan_learning_rate = None
        # Learning rate for the analyzer optimizer.
        self.analyzer_learning_rate = None

    def initialize(self):
        """
    Initialize the class.
    """

        # Initialize neural network models.
        self.modelG = GeneratorNetwork(input_shape=self.noise_dim, output_shape=self.sut.ndimensions,
                                       neurons=self.gan_neurons).to(self.device)
        self.modelC = CriticNetwork(input_shape=self.sut.ndimensions,
                                    neurons=self.gan_neurons).to(self.device)

        # Initialize the analyzer.
        self.analyzer = Analyzer_NN_weighted_new(self.sut.ndimensions, self.device, self.logger)
        self.analyzer.learning_rate = self.analyzer_learning_rate
        self.analyzer.neurons = self.analyzer_neurons
        self.analyzer.initialize()

        # Optimizers.
        self.optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=self.gan_learning_rate, betas=(0, 0.9))
        self.optimizerC = torch.optim.Adam(self.modelC.parameters(), lr=self.gan_learning_rate, betas=(0, 0.9))

    def save(self, identifier, path):
        """
    Save the model to the given path. Files critic_{identifier},
    generator_{identifier}, and analyzer_{identifier} are created in the
    directory.
    """

        torch.save(self.modelC.state_dict(), os.path.join(path, "critic_{}".format(identifier)))
        torch.save(self.modelG.state_dict(), os.path.join(path, "generator_{}".format(identifier)))
        self.analyzer.save(identifier, path)

    def load(self, identifier, path):
        """
    Load the model from path. Files critic_{identifier},
    generator_{identifier}, and analyzer_{identifier} are expected to exist.
    """

        c_file_name = os.path.join(path, "critic_{}".format(identifier))
        g_file_name = os.path.join(path, "generator_{}".format(identifier))

        if not os.path.exists(c_file_name):
            raise Exception("File '{}' does not exist in {}.".format(c_file_name, path))
        if not os.path.exists(g_file_name):
            raise Exception("File '{}' does not exist in {}.".format(g_file_name, path))

        self.modelC.load_state_dict(torch.load(c_file_name))
        self.modelG.load_state_dict(torch.load(g_file_name))
        self.modelC.eval()
        self.modelG.eval()
        self.analyzer.load(identifier, path)

    def train_analyzer_with_batch(self, data_X, data_Y, train_settings, log=False):
        """
    Train the analyzer part of the model with a batch of training data.
    Args:
      data_X (np.ndarray):   Array of tests of shape (N, self.sut.ndimensions).
      data_Y (np.ndarray):   Array of test outputs of shape (N, 1).
      train_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:
                               analyzer_epochs: How many total runs are made
                               with the given training data.
                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

        self.analyzer.train_with_batch(data_X, data_Y, train_settings, log=log)

    def train_with_batch(self, data_X, data_Y, train_settings, log=False):
        """
    Train the WGAN with a batch of training data.
    Args:
      data_X (np.ndarray):   Array of tests of shape (M, self.sut.ndimensions).
      data_Y (np.ndarray):   Array of test outputs of shape (M, 1).
      train_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:
                               critic_epochs: How many times the critic is
                               trained per epoch.
                               generator_epochs: How many times the generator
                               is trained per epoch.
                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

        if len(data_X.shape) != 2 or data_X.shape[1] != self.sut.ndimensions:
            raise ValueError("Array data_X expected to have shape (N, {}).".format(self.ndimensions))
        if len(data_Y.shape) != 2 or data_Y.shape[0] < data_X.shape[0]:
            raise ValueError("Array data_Y array should have at least as many elements as there are tests.")

        data_X = torch.from_numpy(data_X).float().to(self.device)
        data_Y = torch.from_numpy(data_Y).float().to(self.device)

        # Unpack values from the epochs dictionary.
        critic_epochs = train_settings["critic_epochs"] if "critic_epochs" in train_settings else 1
        generator_epochs = train_settings["generator_epochs"] if "generator_epochs" in train_settings else 1

        # Save the training modes for later restoring.
        training_C = self.modelC.training
        training_G = self.modelG.training

        # Train the critic.
        # -----------------------------------------------------------------------
        self.modelC.train(True)
        for m in range(critic_epochs):
            # Here the mini batch size of the WGAN-GP is set to be the number of
            # training samples for the critic
            M = data_X.shape[0]

            # Loss on real data.
            real_inputs = data_X
            real_outputs = self.modelC(real_inputs)
            real_loss = real_outputs.mean(0)

            # Loss on generated data.
            # For now we use as much generated data as we have real data.
            noise = ((torch.rand(size=(M, self.modelG.input_shape)) - 0.5) / 0.5).to(self.device)
            fake_inputs = self.modelG(noise)
            fake_outputs = self.modelC(fake_inputs)
            fake_loss = fake_outputs.mean(0)

            # Gradient penalty.
            # Compute interpolated data.
            e = torch.rand(size=(M, 1)).to(self.device)
            interpolated_inputs = e * real_inputs + (1 - e) * fake_inputs
            # Get critic output on interpolated data.
            interpolated_outputs = self.modelC(interpolated_inputs)
            # Compute the gradients wrt to the interpolated inputs.
            # Warning: Showing the validity of the following line requires some pen
            #          and paper calculations.
            gradients = torch.autograd.grad(inputs=interpolated_inputs,
                                            outputs=interpolated_outputs,
                                            grad_outputs=torch.ones_like(interpolated_outputs).to(self.device),
                                            create_graph=True,
                                            retain_graph=True)[0]

            # We add epsilon for stability.
            epsilon = 0.000001
            gradients_norms = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)
            gradient_penalty = ((gradients_norms - 1) ** 2).mean()

            C_loss = fake_loss - real_loss + self.gp_coefficient * gradient_penalty
            self.optimizerC.zero_grad()
            C_loss.backward()
            self.optimizerC.step()

            if log:
                self.log("Critic epoch {}/{}, Loss: {}, GP: {}".format(m + 1, critic_epochs, C_loss[0],
                                                                       self.gp_coefficient * gradient_penalty))

        self.modelC.train(False)

        # Visualize the computational graph.
        # print(make_dot(C_loss, params=dict(self.modelC.named_parameters())))

        # Train the generator.
        # -----------------------------------------------------------------------
        self.modelG.train(True)
        for m in range(generator_epochs):
            # For now we use as much generated data as we have real data.
            noise = ((torch.rand(size=(data_X.shape[0], self.modelG.input_shape)) - 0.5) / 0.5).to(self.device)
            outputs = self.modelC(self.modelG(noise))

            G_loss = -outputs.mean(0)
            self.optimizerG.zero_grad()
            G_loss.backward()
            self.optimizerG.step()

            if log:
                self.log("Generator epoch {}/{}, Loss: {}".format(m + 1, generator_epochs, G_loss[0]))

        self.modelG.train(False)

        if log:
            # Same as above in critic training.
            real_inputs = data_X
            real_outputs = self.modelC(real_inputs)
            real_loss = real_outputs.mean(0)

            # For now we use as much generated data as we have real data.
            noise = ((torch.rand(size=(real_inputs.shape[0], self.modelG.input_shape)) - 0.5) / 0.5).to(self.device)
            fake_inputs = self.modelG(noise)
            fake_outputs = self.modelC(fake_inputs)
            fake_loss = fake_outputs.mean(0)

            W_distance = real_loss - fake_loss

            self.log("Batch W. distance: {}".format(W_distance[0]))

        # Visualize the computational graph.
        # print(make_dot(G_loss, params=dict(self.modelG.named_parameters())))

        # Restore the training modes.
        self.modelC.train(training_C)
        self.modelG.train(training_G)

    def generate_test(self, N=1):
        """
    Generate N random tests.
    Args:
      N (int): Number of tests to be generated.
    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

        if N <= 0:
            raise ValueError("The number of tests should be positive.")

        training_G = self.modelG.training
        # Generate uniform noise in [-1, 1].
        noise = ((torch.rand(size=(N, self.noise_dim)) - 0.5) / 0.5).to(self.device)
        self.modelG.train(False)
        result = self.modelG(noise).cpu().detach().numpy()
        self.modelG.train(training_G)
        return result

    def predict_fitness(self, test):
        """
    Predicts the fitness of the given test.
    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).
    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

        return self.analyzer.predict(test)
