import numpy as np

class NnModel:
    """
    Base class for training models and passing logs to other classes

    :param input: path to the file with matrices, which are inputs for NN
    :param output: path to the file with matrices, which are outputs for NN
    """

    def __init__(self, input, output):
        self.train_metadata = {}
        self.input = input
        self.output = output

    def train_single_model(self):
        pass

    def train_several_models(self):
        pass

    def _load_from_zip(self):
        pass

    def _save_to_zip(self):
        pass
