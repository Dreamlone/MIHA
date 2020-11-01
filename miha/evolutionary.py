import torch
from torch import jit

def generate_population(actual_opt_path, actual_model_path, temporary_path, amount_of_individuals = 4):
    """
    Method for generating a population

    :param actual_opt_path: path to the file with the current state of the optimizer
    :param actual_model_path: path to the file with the current state of the neural network
    :param temporary_path: the temporary folder where are stored the trained models
    :param amount_of_individuals: number of individuals required
    """

    # Load models
    actual_model = jit.load(actual_model_path)
    optimizer_state = torch.load(actual_opt_path)

    print('I generate population!')


def crossover():
    print('I make crossover!')