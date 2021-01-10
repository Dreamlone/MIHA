import os
import shutil
import torch
from torch import jit
import torch.nn as nn
import torch.utils.data as data_utils
from copy import deepcopy

from miha.evolutionary import *
from miha.log import ModelLogger, PopulationLogger, get_device


class NNOptimizer:
    """
    Base class for configuring hyperparameters and training the neural network

    :param nn_type: neural network architecture for optimization
    (available types 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param input: path to the file with matrices, which are inputs for NN
    :param output: path to the file with matrices, which are outputs for NN
    :param cycles: number of cycles to optimize
    :param population_size: number of neural networks at the population generation stage
    :param epoch_per_cycle: how many epochs should the neural network be trained
    after the crossover in each cycle
    :param fixing_epochs: how many epochs should a single model be trained in
    one cycle after crossover runup_epochs
    :param runup_epochs: the number of epochs to complete before entering the
    loop cycles and after the optimization is complete
    :param save_logs: do we need to save logs
    :param logs_folder: path to th folder where do we need to save logs, ignore
    when save_logs = False
    """

    def __init__(self, nn_type: str, task: str, input: str, output: str,
                 cycles: int = 2, population_size: int = 4, epoch_per_cycle: int = 2,
                 fixing_epochs: int = 4, runup_epochs: int = 3,
                 save_logs: bool = False, logs_folder: str = None):
        self.nn_type = nn_type
        self.task = task
        self.input = input
        self.output = output
        self.cycles = cycles
        self.save_logs = save_logs
        self.population_size = population_size

        # The number of epochs required for initial training up to
        # the population generation stage
        self.runup_epochs = runup_epochs

        self.epoch_per_cycle = epoch_per_cycle
        self.fixing_epochs = fixing_epochs

        if self.save_logs == True:
            self.logs_folder = logs_folder
        else:
            # Create temporary folder
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.logs_folder = os.path.join(dir_path, 'tmp_folder')

        # Create folder if it doesnt exists
        if os.path.isdir(self.logs_folder) == False:
            os.makedirs(self.logs_folder)

        # Init model logger with log folder (where to save models)
        self.logger = ModelLogger(logs_path=self.logs_folder,
                                  nn_type=self.nn_type)

        # Init logger for population of models and evolutionary algorithm
        self.pop_logger = PopulationLogger(logs_path=self.logs_folder,
                                           nn_type=self.nn_type,
                                           pop_size=self.population_size,
                                           cycles=self.cycles,
                                           epoch_per_cycle=self.epoch_per_cycle)

    def optimize(self, source_nn, source_loss, source_optimizer,
                 source_batch_size: int = 32) -> dict:
        """
        A method for finding the optimal set of hyperparameters for a given architecture

        :param source_nn: class with the specified initial neural network architecture
        :param source_loss: initial loss function
        :param source_optimizer: initial optimizer
        :param source_batch_size: batch size

        :return : dictionary with obtained model
            - model: neural network model
            - loss: loss function
            - optimizer: obtained optimizer
            - batch: batch size
        """

        print('Init cycle')
        self.current_cycle = 0

        # Initial configurations of the model, loss function, and optimizer
        self.source_nn_class = source_nn
        self.current_nn = source_nn()
        self.current_criterion = source_loss()
        self.current_optimizer = source_optimizer(self.current_nn.parameters())
        self.current_batch_size = source_batch_size

        # Initial train for n amount of epochs
        self._train(n_epochs=self.runup_epochs)

        ######################
        # Start optimization #
        ######################
        for i in range(1, self.cycles + 1):
            self.current_cycle = i
            print(f'\nOptimizing cycle number {self.current_cycle}')

            # Get actual model and optimizator path
            actual_model_path = self.logger.get_actual_model_path()
            actual_opt_path = self.logger.get_actual_opt_path()
            actual_model_pth_path = self.logger.get_actual_model_pth_path()

            # Get population - list [NN_0, NN_1, ..., NN_self.population_size]
            # And description of changes in chgs_list
            nns_list, chgs_list = generate_population(nn_type=self.nn_type,
                                                      task=self.task,
                                                      actual_model_path=actual_model_path,
                                                      actual_opt_path=actual_opt_path,
                                                      actual_model_pth_path=actual_model_pth_path,
                                                      source_nn_class=self.source_nn_class,
                                                      actual_optimizer=self.current_optimizer,
                                                      actual_criterion=self.current_criterion,
                                                      actual_batch_size=self.current_batch_size,
                                                      amount_of_individuals=self.population_size)

            # Train each individual by _train_population
            self._train_population(nns_list, chgs_list)

            # Calculate fitness score for every NN in nns_list
            pop_cycle_metadata = self.pop_logger.get_metadata(cycle=self.current_cycle)
            fitness_list = eval_fitness(metadata=pop_cycle_metadata)

            print(f'Fitness list: {fitness_list}')

            # Crossover -> get NN to continue training
            updated_model = crossover(fitness_list, nns_list)
            self.current_nn = updated_model['model']
            self.current_criterion = updated_model['loss']
            self.current_optimizer = updated_model['optimizer']
            self.current_batch_size = updated_model['batch']

            self._train(n_epochs=self.fixing_epochs)
        #######################
        # Finish optimization #
        #######################

        # Finish training model with final training set of epochs
        print('\nFinal cycle')
        self.current_cycle = -1
        self._train(n_epochs=self.runup_epochs)

        # Save "population metadata" to txt file
        self.pop_logger.save_metadata()

        # Plot training
        self.logger.plot_scores()
        if self.save_logs == False:
            # Deleting the temporary directory
            shutil.rmtree(self.logs_folder, ignore_errors=True)

        return {'model':self.current_nn, 'loss': self.current_criterion,
                'optimizer':self.current_optimizer, 'batch': self.current_batch_size}

    def _train(self, n_epochs: int) -> None:
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

        self.current_nn = self.current_nn.to(device)
        self.current_nn = self.current_nn.train(mode=True)

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

    def _train_population(self, nns_list: list, changes_list: list) -> None:
        """
        Method for training a population (several NN)

        :param nns_list: list with neural network models as dict, where
            - model: neural network model
            - loss: loss function
            - optimizer: obtained optimizer
            - batch: batch size
        :param changes_list: list with descriptions of changes
        """

        # Train model on GPU
        device = get_device()

        for model_number, nn_to_train in enumerate(nns_list):
            model_to_train = nn_to_train['model']
            loss_to_train = nn_to_train['loss']
            optimizer_to_train = nn_to_train['optimizer']
            batch_to_train = nn_to_train['batch']

            # Load X and Y matrices in tensors
            x_train = torch.load(self.input)
            y_train = torch.load(self.output)

            train = data_utils.TensorDataset(x_train, y_train)

            # Prepare data loaders
            train_loader = torch.utils.data.DataLoader(train,
                                                       batch_size=batch_to_train,
                                                       num_workers=0)

            for epoch in range(1, self.epoch_per_cycle + 1):
                train_loss = 0.0

                # Training
                for arrays, target in train_loader:
                    # Go this matrices to GPU
                    arrays = arrays.to(device)
                    target = target.to(device)

                    # Refresh gradients
                    optimizer_to_train.zero_grad()

                    # Outputs from NN
                    outputs = model_to_train(arrays)

                    # Measure the difference between the actual values and the prediction
                    loss = loss_to_train(outputs, target)
                    loss.backward()
                    optimizer_to_train.step()
                    train_loss += loss.item() * target.size(0)

                train_loss = train_loss / len(train_loader)

                # Log scores after model training
                self.pop_logger.collect_scores(model_number=model_number,
                                               current_cycle=self.current_cycle,
                                               current_epoch=epoch,
                                               model_score=train_loss,
                                               change=changes_list[model_number])

            # Save trained model
            self.pop_logger.save_nn(current_cycle=self.current_cycle,
                                    model_number=model_number,
                                    nn_model=model_to_train,
                                    nn_optimizer=optimizer_to_train,
                                    nn_loss=loss_to_train,
                                    nn_batch=batch_to_train)
