import os
import shutil
import torch
from torch import jit
import torch.utils.data as data_utils
from miha.evolutionary import *
from miha.log import ModelLogger


class NNOptimizer:
    """
    Base class for configuring hyperparameters and training the neural network

    :param nn_type: neural network architecture for optimization
    (available types 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param input: path to the file with matrices, which are inputs for NN
    :param output: path to the file with matrices, which are outputs for NN
    :param cycles: number of cycles to optimize
    :param population_size: number of neural networks at the population generation stage
    :param epoch_per_cycle: how many epochs should the neural network be trained
    after the crossover in each cycle
    :param save_logs: do we need to save logs
    :param logs_folder: path to th folder where do we need to save logs, ignore when save_logs = False
    """

    def __init__(self, nn_type, input, output, cycles=2, population_size=4,
                 epoch_per_cycle=2, save_logs=False, logs_folder=None):
        self.nn_type = nn_type
        self.input = input
        self.output = output
        self.cycles = cycles
        self.save_logs = save_logs
        self.population_size = population_size

        # The number of epochs required for initial training up to
        # the population generation stage
        # TODO determine amount of init epochs - const(2? 10? 50?) or variable
        self.init_epochs = 3

        self.epoch_per_cycle = epoch_per_cycle

        if self.save_logs == True:
            self.logs_folder = logs_folder
        else:
            # Create temporary folder
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.logs_folder = os.path.join(dir_path, 'tmp_folder')

        # Create folder if it doesnt exists
        if os.path.isdir(self.logs_folder) == False:
            os.makedirs(self.logs_folder)

        # Init logger with log folder (where to save models)
        self.logger = ModelLogger(logs_path=self.logs_folder,
                                  nn_type=self.nn_type)

    def optimize(self, source_nn, source_loss, source_optimizer, source_batch_size = 32):
        """
        A method for finding the optimal set of hyperparameters for a given architecture

        :param source_nn: class with the specified initial neural network architecture
        :param source_loss: initial loss function
        :param source_optimizer: initial optimizer
        :param source_batch_size: batch size

        :return : словарь с
        """

        print('Init cycle')
        self.current_cycle = 0

        # Initial configurations of the model, loss function, and optimizer
        self.current_nn = source_nn()
        self.current_criterion = source_loss()
        self.current_optimizer = source_optimizer(self.current_nn.parameters())
        self.current_batch_size = source_batch_size

        # Initial train for n amount of epochs
        self._train(n_epochs = self.init_epochs)

        ######################
        # Start optimization #
        ######################
        for i in range(1, self.cycles + 1):
            self.current_cycle = i
            print(f'\nOptimizing cycle number {self.current_cycle}')

            # Get paths to the best NN model at the current moment
            actual_opt_path = self.logger.get_actual_opt_path()
            actual_model_path = self.logger.get_actual_model_path()

            # Get population - list [NN_1, NN_2, ..., NN_self.population_size]
            nns_list = generate_population(actual_opt_path=actual_opt_path,
                                           actual_model_path=actual_model_path,
                                           amount_of_individuals=self.population_size)

            # Train each individual by _train_population
            self._train_population(nns_list)

            # Calculate fitness score for every NN in nns_list
            fitness_list = eval_fitness(nns_list)

            # Crossover -> get NN to continue training
            updated_model = crossover(fitness_list, nns_list)
            self.current_nn = updated_model['model']
            self.current_criterion = updated_model['loss']
            self.current_optimizer = updated_model['optimizer']
            self.current_batch_size = updated_model['batch']

            self._train(n_epochs=self.epoch_per_cycle)
        #######################
        # Finish optimization #
        #######################

        # Plot training
        self.logger.plot_scores()
        if self.save_logs == False:
            # Deleting the temporary directory
            shutil.rmtree(self.logs_folder, ignore_errors=True)

        return {'model':self.current_nn, 'loss': self.current_criterion,
                'optimizer':self.current_optimizer, 'batch': self.current_batch_size}

    def _train(self, n_epochs):
        """
        Method for training a neural network in the "model" module.
        All actions are mutable for NN, so train work "in place".

        :param n_epochs: number of epochs to train
        """

        # Load X and Y matrices in tensors
        x_train = torch.load(self.input)
        y_train = torch.load(self.output)

        train = data_utils.TensorDataset(x_train, y_train)

        # Prepare data loaders
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.current_batch_size, num_workers=0)

        # Replace NN to GPU
        device = get_device()
        self.current_nn.to(device)

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0

            # Training
            for arrays, target in train_loader:
                # Go this matrices to GPU
                arrays = arrays.to(device)
                target = target.to(device)

                # Refresh gradients
                self.current_optimizer.zero_grad()

                # Outputs from NN
                outputs = self.current_nn(arrays)
                # Measure the difference between the actual values and the prediction
                loss = self.current_criterion(outputs, target)
                loss.backward()
                self.current_optimizer.step()
                train_loss += loss.item() * target.size(0)

            train_loss = train_loss / len(train_loader)

            # Update logs and save fitted model
            if epoch == n_epochs:
                is_last_epoch = True
                # Update model in logger
                self.logger.set_current_model(nn_model=self.current_nn,
                                              nn_optimizer=self.current_optimizer)
            else:
                is_last_epoch = False
            self.logger.update_history(current_cycle=self.current_cycle,
                                       current_epoch=epoch,
                                       is_last_epoch=is_last_epoch,
                                       model_score=train_loss)

            print('Epoch: {} \tTraining Loss: {:.2f}'.format(epoch, train_loss))

    def _train_population(self, nns_list):
        """
        Method for training a population (several NN)

        :param nns_list: list with neural network models
        """

        pass

def get_device():
    """
    Method for getting available devices.

    """

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device