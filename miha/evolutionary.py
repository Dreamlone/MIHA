from miha.log import PopulationLogger, get_device
from copy import deepcopy, copy
from torch import jit
import torch


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


def generate_population(nn_type: str, actual_model_path: str, actual_opt_path: str,
                        actual_criterion, actual_batch_size, amount_of_individuals: int) -> list:
    """
    Method for generating a population

    :param nn_type: the type of NN architecture (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param actual_model_path: path to zip file with current NN model
    :param actual_opt_path: path to pth optimizer of current NN model
    :param actual_criterion: loss of current NN model
    :param actual_batch_size: current batch size
    :param amount_of_individuals: number of individuals required

    :return nns_list: list with neural network models as dict, where
        - model: neural network model
        - loss: loss function
        - optimizer: obtained optimizer
        - batch: batch size
    """

    # Define converter for EA NN model representation
    # ea_converter = EARepresentation(nn_type, actual_model, actual_criterion,
    #                                 actual_optimizer, actual_batch_size)
    # actual_model_ea_form = ea_converter.convert_to_genotype()

    nns_list = []
    for i in range(0, amount_of_individuals):
        actual_model = jit.load(actual_model_path)
        state = torch.load(actual_opt_path)

        device = get_device()
        actual_model = actual_model.to(device)
        actual_model = actual_model.train(mode=True)

        # TODO make optimizer adaptive
        actual_optimizer = torch.optim.Adam(actual_model.parameters())
        actual_optimizer.load_state_dict(state['optimizer'])

        # Make copies for all parameters
        criterion_copy = deepcopy(actual_criterion)
        batch_copy = deepcopy(actual_batch_size)

        nns_list.append({'model': actual_model, 'loss': criterion_copy,
                         'optimizer': actual_optimizer, 'batch': batch_copy})

    return nns_list


def eval_fitness(metadata: dict) -> list:
    """
    Function for evaluating the progress of multiple neural networks during training

    :param metadata: metadata about the population for a particular cycle in the
    form of a dictionary
        - key: model - index of the model (neural network)
        - value: list [a, b], where a - list with loss scores [....] and b -
        verbal description of what replacement was made in the neural network
        during mutation

    :return fitness_list: list with fitness scores
    """

    # Get metadata f
    models = list(metadata.keys())
    models.sort()

    # Calculate loss diff per epoch
    fitness_list = []
    for model in models:
        model_info = metadata.get(model)

        # Scores array with train losses
        scores_arr = model_info[0]

        # Calculate efficiency of NN
        fitness = scores_arr[0]-scores_arr[-1]
        fitness_list.append(fitness)

    return fitness_list


def crossover(fitness_list, nns_list):

    nn_model = nns_list[0]
    return nn_model
