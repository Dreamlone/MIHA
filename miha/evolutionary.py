from miha.log import get_device
import numpy as np
from copy import deepcopy
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

        if new_batch_size != self.batch_size:
            description = ''.join(('batch size was changed from ',
                                   str(self.batch_size),' to ', str(new_batch_size)))
        else:
            description = 'nothing has changed'

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': new_batch_size}

        return model_dict, description

    def change_loss_criterion(self):
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change loss criterion'

        return model_dict, description

    def change_layer_activations(self):
        """
        The method allows you to replace the activation function in the selected
         neural network layer

        """

        # Get layer names
        layer_names = []
        for layer_name, _ in self.model.named_children():
            layer_names.append(layer_name)

        amount_layers = len(layer_names)

        # Randomly choose layer by index
        layer_index = random.randint(0, amount_layers-1)
        random_layer = layer_names[layer_index]

        # Get available activations functions
        if layer_index == (amount_layers-1):
            act_functions = self._get_available_activations(is_this_layer_last=True)
        else:
            act_functions = self._get_available_activations(is_this_layer_last=False)

        # Randomly choose activation function (get name of it)
        new_name_function = random.choice(act_functions)

        # Get activation function object
        new_act_function = self._get_act_by_name(new_name_function)

        # Make changes
        self.model, prev_act = self._convert_activations(model=self.model,
                                                         layer_index=layer_index,
                                                         new_act_function=new_act_function)

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}

        if prev_act == new_name_function:
            description = 'nothing has changed'
        else:
            description = ''.join(('in layer with name ', str(random_layer),
                                   ' was changed activation function from ',
                                   prev_act, ' to ', new_name_function))

        return model_dict, description

    def change_neurons_activations(self):
        """
        TODO implement

        """
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change layer neuron functions'

        return model_dict, description

    def change_optimizer(self):
        """
        The method changes the optimizer of NN model

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

    def _get_available_activations(self, is_this_layer_last: bool) -> list:
        """
        The method returns the available activation functions for the selected task

        :param is_this_layer_last: is the layer for which activation functions
        are selected the final layer

        :return activations_list: list with names of an appropriate activation
        functions
        """

        if is_this_layer_last == True:
            if self.task == 'regression':
                activations_list = ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh',
                                    'ReLU', 'ReLU6', 'SELU', 'Sigmoid', 'Tanh']
            elif self.task == 'classification':
                activations_list = ['Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
        else:
            activations_list = ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh',
                                'ReLU', 'ReLU6', 'SELU', 'Sigmoid', 'Tanh']
        return activations_list

    def _convert_activations(self, model, layer_index, new_act_function):
        """
        The method replaces the activation function in the selected layer

        :param model: NN model to process
        :param layer_index: index of the layer where you want to replace the
        activation function
        :param name_function: activation function to be replaced

        :return model: the model is replaced by the activation function
        :return prev_act: name of the previous activation function in the layer
        TODO there is a need to make the function more adaptive
        """

        current_index = 0
        for number_1, child_1 in model.named_children():
            # We achieve appropriate layer
            if current_index == layer_index:
                for number_2, child_2 in child_1.named_children():

                    # Find the second layer with activation function
                    if int(number_2) == 1:
                        prev_act = str(child_2)
                        prev_act = prev_act[:-2]
                        # Set new activation function
                        setattr(child_1, number_2, new_act_function)
                        break
            current_index += 1

        return model, prev_act

    def _get_act_by_name(self, name_function):
        """
        TODO need to refactor
        The method returns the corresponding function by it's name

        :param name_function: name of function
        :return fucntion_obj: new activation function
        """

        if name_function == 'ELU':
            fucntion_obj = torch.nn.ELU()
        elif name_function == 'Hardshrink':
            fucntion_obj = torch.nn.Hardshrink()
        elif name_function == 'Hardsigmoid':
            fucntion_obj = torch.nn.Hardsigmoid()
        elif name_function == 'Hardtanh':
            fucntion_obj = torch.nn.Hardtanh()
        elif name_function == 'ReLU':
            fucntion_obj = torch.nn.ReLU()
        elif name_function == 'ReLU6':
            fucntion_obj = torch.nn.ReLU6()
        elif name_function == 'SELU':
            fucntion_obj = torch.nn.SELU()
        elif name_function == 'Sigmoid':
            fucntion_obj = torch.nn.Sigmoid()
        elif name_function == 'Tanh':
            fucntion_obj = torch.nn.Tanh()
        elif name_function == 'Softmin':
            fucntion_obj = torch.nn.Softmin()
        elif name_function == 'Softmax':
            fucntion_obj = torch.nn.Softmax()
        elif name_function == 'Softmax2d':
            fucntion_obj = torch.nn.Softmax2d()
        elif name_function == 'LogSoftmax':
            fucntion_obj = torch.nn.LogSoftmax()

        return fucntion_obj


def generate_population(nn_type: str, task: str, actual_model_path: str, actual_opt_path: str,
                        actual_model_pth_path, source_nn_class, actual_optimizer,
                        actual_criterion, actual_batch_size, amount_of_individuals: int, act_mod):
    """
    Method for generating a population

    :param nn_type: the type of NN architecture (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param actual_model_path: path to zip file with current NN model
    :param actual_opt_path: path to pth optimizer of current NN model
    :param actual_model_pth_path: path to NN model as pth file
    :param source_nn_class: class for NN init
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
        state = torch.load(actual_opt_path)

        actual_model = deepcopy(act_mod)

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
        # TODO implement change_loss_criterion, change_neurons_activations operators
        operators = ['change_batch_size',
                     'change_layer_activations',
                     'change_optimizer']

        random_operator = random.choice(operators)
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
    """
    For now it returns only 1 best model without crossover (only selection)
    TODO implement crossover operator in right way
    """

    # If there is only 1 model in population
    if len(fitness_list) == 1:
        nn_model = nns_list[0]
    else:
        # Selection: choose 2 the fittest models
        fitness_list = np.array(fitness_list)
        best_ids = np.argmax(fitness_list)
        nn_model = nns_list[best_ids]
    return nn_model
