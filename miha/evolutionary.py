import torch
from torch import jit


class EARepresentation:
    """
    A class that allows you to represent neural networks in a form that is
    convenient for the evolutionary algorithm

    :param nn_type: the type of architecture that you want to represent in
    encoded form (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    """

    def __init__(self, nn_type: str):
        self.nn_type = nn_type

    def convert_to_genotype(self):
        pass

    def convert_from_genotype(self):
        pass


def generate_population(actual_opt_path, actual_model_path, amount_of_individuals):
    """
    Method for generating a population

    :param actual_opt_path: path to the file with the current state of the optimizer
    :param actual_model_path: path to the file with the current state of the neural network
    :param amount_of_individuals: number of individuals required

    :return nns_list: list with neural network models
    """

    # Load model
    actual_model = jit.load(actual_model_path)
    optimizer_state = torch.load(actual_opt_path)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in actual_model.state_dict():
        print(param_tensor, "\t", actual_model.state_dict()[param_tensor].size())

    print()

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer_state:
        print(var_name, "\t", optimizer_state[var_name].size())

    nns_list = [0,0,0,0]
    return nns_list

def eval_fitness(nns_list):

    fitness_list = [0, 0, 0, 0]
    return fitness_list

def crossover(fitness_list, nns_list):

    nn_model = {'model': [0], 'loss': [0], 'optimizer': [0], 'batch': [0]}
    return nn_model