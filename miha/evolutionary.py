from miha.log import get_device
from copy import deepcopy, copy
from torch import jit
import torch
import random


class Mutator:
    """
    Class for performing the mutation procedure

    :param nn_type: the type of NN architecture that you want to represent in
    encoded form (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param model: current NN model
    :param criterion: loss of current NN model
    :param optimizer: optimizer of current NN model
    :param batch_size: current batch size
    """

    def __init__(self, nn_type: str, task: str, model, criterion,
                 optimizer, batch_size):
        self.nn_type = nn_type
        self.task = task

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    def change_batch_size(self):
        """
        The method changes the size of the batch by a certain value

        :return model_dict: dictionary with model
        :return description: descriptions of changes
        """

        # Add a new number to current value
        new_batch_size = self.batch_size + random.randint(-10, 10)

        # Checking the validity of the solution
        if new_batch_size > 0:
            if new_batch_size < 200:
                pass
            else:
                new_batch_size = 200
        else:
            new_batch_size = self.batch_size + 2

        description = ''.join(('batch size was changed from ', str(self.batch_size),
                               ' to ', str(new_batch_size)))

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': new_batch_size}

        return model_dict, description

    def change_loss_criterion(self):
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change loss criterion'

        return model_dict, description

    def change_layer_activations(self):
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change layer activation functions'

        return model_dict, description

    def change_neurons_activations(self):
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change layer neuron functions'

        return model_dict, description

    def change_optimizer(self):
        """
        The method changes the optimizer of NN  model

        :return model_dict: dictionary with model
        :return description: descriptions of changes
        """

        # Find out the name of the optimizer
        current_opt = str(self.optimizer.__class__)
        current_opt = current_opt.split("'")[-2]
        # One more split
        current_opt = current_opt.split(".")[-1]

        # Randomly select the optimizer
        available_optimizers = ['SGD', 'AdamW', 'Adam', 'Adadelta']
        random_optimizer = random.choice(available_optimizers)

        if random_optimizer == 'SGD' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to SGD'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.00001, momentum=0.9)
        elif random_optimizer == 'AdamW' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to AdamW'))
            self.optimizer = torch.optim.AdamW(self.model.parameters())
        elif random_optimizer == 'Adam' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to Adam'))
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif random_optimizer == 'Adadelta' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to Adadelta'))
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        else:
            description = 'nothing has changed'
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}

        return model_dict, description

def generate_population(nn_type: str, task: str, actual_model_path: str, actual_opt_path: str,
                        actual_optimizer, actual_criterion, actual_batch_size, amount_of_individuals: int):
    """
    Method for generating a population

    :param nn_type: the type of NN architecture (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param actual_model_path: path to zip file with current NN model
    :param actual_opt_path: path to pth optimizer of current NN model
    :param actual_optimizer: current optimizer
    :param actual_criterion: loss of current NN model
    :param actual_batch_size: current batch size
    :param amount_of_individuals: number of individuals required

    :return nns_list: list with neural network models as dict, where
        - model: neural network model
        - loss: loss function
        - optimizer: obtained optimizer
        - batch: batch size
    :return changes_list: list with descriptions of changes
    """

    nns_list = []
    changes_list = []
    for i in range(0, amount_of_individuals):
        actual_model = jit.load(actual_model_path)
        state = torch.load(actual_opt_path)

        device = get_device()
        actual_model = actual_model.to(device)
        actual_model = actual_model.train(mode=True)

        # Optimizer adaptive
        optimizer_class = actual_optimizer.__class__
        # Different optimizers need different initialisation
        if optimizer_class == torch.optim.SGD:
            actual_optimizer = optimizer_class(actual_model.parameters(),
                                               lr=0.0001, momentum=0.9)
        else:
            actual_optimizer = optimizer_class(actual_model.parameters())
        actual_optimizer.load_state_dict(state['optimizer'])

        # Make copies for all parameters
        criterion_copy = deepcopy(actual_criterion)
        batch_copy = deepcopy(actual_batch_size)

        # Define mutation operator class
        mut_operator = Mutator(nn_type=nn_type,
                               task=task,
                               model=actual_model,
                               criterion=criterion_copy,
                               optimizer=actual_optimizer,
                               batch_size=batch_copy)

        # Make mutation
        operators = ['change_batch_size', 'change_loss_criterion',
                     'change_layer_activations', 'change_neurons_activations',
                     'change_optimizer']

        random_operator = random.choice(operators)
        # TODO remove it ->
        random_operator = 'change_layer_activations'
        if random_operator == 'change_batch_size':
            mutated_model, change = mut_operator.change_batch_size()
        elif random_operator == 'change_loss_criterion':
            mutated_model, change = mut_operator.change_loss_criterion()
        elif random_operator == 'change_layer_activations':
            mutated_model, change = mut_operator.change_layer_activations()
        elif random_operator == 'change_neurons_activations':
            mutated_model, change = mut_operator.change_neurons_activations()
        elif random_operator == 'change_optimizer':
            mutated_model, change = mut_operator.change_optimizer()

        nns_list.append(mutated_model)
        changes_list.append(change)

    return nns_list, changes_list


def eval_fitness(metadata: dict) -> list:
    """
    Function for evaluating the progress of multiple neural networks during training

    :param metadata: metadata about the population for a particular cycle in the
    form of a dictionary
        - key: model - index of the model (neural network)
        - value: list [a, b], where a - list with loss scores [...] and b -
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
