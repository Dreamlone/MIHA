import numpy as np
from miha.model import NnModel

class CnnAutoencoder:
    """
    Base class for configuring hyperparameters and training the neural network architecture
    convolutional autoencoder

    :param save_logs: do we need to save logs
    """

    def __init__(self, save_logs = False):
        self.metadata = {}
        self.save_logs = save_logs


    def optimize(self, source_nn):
        """
        A method for finding the optimal set of hyperparameters for a given architecture

        :param source_nn: instance of class with the specified initial neural network architecture
        """

        self.source_nn = source_nn

    def _train(self):
        """
        Method for training a neural network in the "model" module

        """

        model_trainier = NnModel()

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