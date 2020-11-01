import numpy as np

class NnModel:
    """
    Base class for training models and passing logs to other classes

    """

    def __init__(self):
        self.train_metadata = {}

    def train_single_model(self):
        pass

    def train_several_models(self):
        pass

    def _load_from_zip(self):
        pass

    def _save_to_zip(self):
        pass
