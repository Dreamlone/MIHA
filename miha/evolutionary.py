import torch
from torch import jit
from miha.log import EALogger, get_device
from copy import copy

class EARepresentation:
    """
    A class that allows you to represent neural networks in a form that is
    convenient for the evolutionary algorithm

    :param nn_type: the type of NN architecture that you want to represent in
    encoded form (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param actual_model: current NN model
    :param actual_criterion: loss of current NN model
    :param actual_optimizer: optimizer of current NN model
    :param actual_batch_size: current batch size
    """

    def __init__(self, nn_type: str, actual_model, actual_criterion,
                 actual_optimizer, actual_batch_size):
        self.nn_type = nn_type

        self.actual_model = actual_model
        self.actual_criterion = actual_criterion
        self.actual_optimizer = actual_optimizer
        self.actual_batch_size = actual_batch_size

    def convert_to_genotype(self):
        return 0

    def convert_from_genotype(self):
        return 0


def generate_population(nn_type, actual_model, actual_criterion,
                        actual_optimizer, actual_batch_size, amount_of_individuals):
    """
    Method for generating a population

    :param nn_type: the type of NN architecture (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param actual_model: current NN model
    :param actual_criterion: loss of current NN model
    :param actual_optimizer: optimizer of current NN model
    :param actual_batch_size: current batch size
    :param amount_of_individuals: number of individuals required

    :return nns_list: list with neural network models
    """
    actual_model.to('cpu')
    actual_model.eval()

    # Define converter for EA NN model representation
    ea_converter = EARepresentation(nn_type, actual_model, actual_criterion,
                                    actual_optimizer, actual_batch_size)
    actual_model_ea_form = ea_converter.convert_to_genotype()

    nns_list = []
    for i in range(0, amount_of_individuals):
        net_copy = copy(actual_model)
        nns_list.append(net_copy)

    return nns_list


def eval_fitness(nns_list):

    fitness_list = [0, 0, 0, 0]
    return fitness_list


def crossover(fitness_list, nns_list):

    nn_model = {'model': nns_list[0], 'loss': [0], 'optimizer': [0], 'batch': [0]}
    return nn_model
