import os
import torch
from miha.model import NnModel

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

    def _get_device(self):
        """
        Method for getting available devices

        """

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def _train(self, n_epochs):
        """
        Method for training a neural network in the "model" module

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

                # Выход нейронной сети
                outputs = self.init_nn(images)
                # Замеряем разницу между действительными значениями и предсказанием
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