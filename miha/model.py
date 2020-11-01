import os
import torch
from torch import jit
from miha.evolutionary import *

class CnnAutoencoder:
    """
    Base class for configuring hyperparameters and training the neural network architecture
    convolutional autoencoder

    :param input: path to the file with matrices, which are inputs for NN
    :param output: path to the file with matrices, which are outputs for NN
    :param cycles: number of cycles to optimize
    :param save_logs: do we need to save logs
    :param logs_folder: path to th folder where do we need to save logs, ignore when save_logs = False
    """

    def __init__(self, input, output, cycles = 2, save_logs = False, logs_folder = None):
        self.metadata = {}

        self.input = input
        self.output = output
        self.cycles = cycles
        self.save_logs = save_logs

        if save_logs == True:
            if os.path.isdir(logs_folder) == False:
                os.makedirs(logs_folder)
            self.logs_folder = logs_folder

        # Create temporary folder
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.tmp_path = os.path.join(dir_path, 'tmp_folder')
        os.makedirs(self.tmp_path)

    def optimize(self, source_nn, source_loss, source_optimizer, source_batch_size = 32):
        """
        A method for finding the optimal set of hyperparameters for a given architecture

        :param source_nn: class with the specified initial neural network architecture
        :param source_loss: initial loss function
        :param source_optimizer: initial optimizer
        :param source_batch_size: batch size
        """

        # Initial configurations of the model, loss function, and optimizer
        self.init_nn = source_nn()
        self.init_criterion = source_loss()
        self.init_optimizer = source_optimizer(self.init_nn.parameters())
        self.source_batch_size = source_batch_size

        # Initial train for n amount of epochs
        self._train(n_epochs = 2)

        # Save model to file
        self.init_nn.to('cpu')
        self.init_nn.eval()

        # Create TorchScript by tracing the computation graph with an example input
        x = torch.ones(16, 1, 2, 2)
        net_trace = jit.trace(self.init_nn, x)
        actual_model_path = os.path.join(self.tmp_path, 'model_init.zip')
        jit.save(net_trace, actual_model_path)

        # Save optimizer configuration to file
        actual_opt_path = os.path.join(self.tmp_path, 'optimizer_init.pth')
        torch.save({'optimizer': self.init_optimizer.state_dict()}, actual_opt_path)

        ######################
        # Start optimization #
        ######################
        for i in range(1, self.cycles + 1):
            print(f'Optimizing cycle number {i}')

            # Get population by _get_population
            # Parameters: actual_opt_path, actual_model_path, self.tmp_path

            # Train each individual by _train_population

            # Calculate fitness

            # Crossover by _merge_individuals


    def _get_device(self):
        """
        Method for getting available devices.

        """

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def _train(self, n_epochs):
        """
        Method for training a neural network in the "model" module.
        All actions are mutable for NN, so train work "in place".

        :param n_epochs: number of epochs to train
        """

        # Load X and Y matrices in tensors
        x_train = torch.load(self.input)
        y_train = torch.load(self.output)

        # Prepare data loaders
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=self.source_batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(y_train, batch_size=self.source_batch_size, num_workers=0)

        # Replace NN to GPU
        device = self._get_device()
        self.init_nn.to(device)

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0

            # Training
            for data in train_loader:
                # Batch with several matrices
                images = data
                # Go this matrices to GPU
                images = images.to(device)
                # Refresh gradients
                self.init_optimizer.zero_grad()

                # Outputs from NN
                outputs = self.init_nn(images)
                # Measure the difference between the actual values and the prediction
                loss = self.init_criterion(outputs, images)
                loss.backward()
                self.init_optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.3f}'.format(epoch, train_loss))

    def _get_population(self, amount_of_individuals = 4):
        """
        Method for generating a population

        :param amount_of_individuals: number of individuals required
        """

        pass

    def _train_population(self):
        """
        Method for training a population (several NN)

        """

        pass

    def _merge_individuals(self):
        """
        Method for combining multiple models into one

        """

        pass