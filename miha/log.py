import os
import torch
import json

import pandas as pd
import numpy as np

from torch import jit
from matplotlib import pyplot as plt


class ModelLogger:
    """
    A class that stores the history of neural network training in the process
    of evolution and perform operations to save models, output messages with logs

    :param nn_type: neural network architecture for optimization
    (available types 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param logs_path: path to the folder where saved versions of the neural network are stored
    """

    def __init__(self, logs_path: str,  nn_type: str):
        self.logs_path = os.path.join(logs_path,'model')
        # Create folder if it doesnt exists
        if os.path.isdir(self.logs_path) == False:
            os.makedirs(self.logs_path)

        self.nn_type = nn_type

        self.train_losses = []
        self.all_cycles = []
        self.all_epochs = []

    def set_current_model(self, nn_model, nn_optimizer) -> None:
        """
        The method updates the current state of the model and optimizer

        :param nn_model: current model
        :param nn_optimizer: current optimizer
        """

        self.nn_model = nn_model
        self.nn_optimizer = nn_optimizer

        # Transfer NN model to CPU
        self.nn_model = self.nn_model.to('cpu')
        self.nn_model.eval()

    def update_history(self, current_cycle: int, current_epoch: int,
                       is_last_epoch: bool, model_score: float) -> None:
        """
        The method updates the log history and saves the current models as archives

        :param current_cycle: current cycle of evolution
        :param current_epoch: in the current epoch when training
        :param is_last_epoch: is this the last epoch in the cycle
        :param model_score: value of the metric in the training sample
        """

        # Update all logs
        self.train_losses.append(model_score)
        self.all_cycles.append(current_cycle)
        self.all_epochs.append(current_epoch)

        # If it's only initialisation NN
        if current_cycle == 0:
            if is_last_epoch == True:
                model_zip = 'model_init.zip'
                model_pth = 'optimizer_init.pth'
                model_path = [os.path.join(self.logs_path, model_zip),
                              os.path.join(self.logs_path, model_pth)]

                # Save NN model
                self._save_nn(model_zip, model_pth)
            else:
                pass
        elif current_cycle == -1:
            if is_last_epoch == True:
                model_zip = 'model_final.zip'
                model_pth = 'optimizer_final.pth'
                model_path = [os.path.join(self.logs_path, model_zip),
                              os.path.join(self.logs_path, model_pth)]

                # Save NN model
                self._save_nn(model_zip, model_pth)
            else:
                pass
        # If there is cycle optimization
        else:
            if is_last_epoch == True:
                model_zip = ''.join(('model_', str(current_cycle), '.zip'))
                model_pth = ''.join(('optimizer_', str(current_cycle), '.pth'))
                model_path = [os.path.join(self.logs_path, model_zip),
                              os.path.join(self.logs_path, model_pth)]

                # Save NN model
                self._save_nn(model_zip, model_pth)
            else:
                pass

    def plot_scores(self) -> None:
        """
        The method allows drawing the values of the error metric at different
        epochs and cycles
        """

        # Let's prepare a dataframe with logs for each cycle and epoch
        df = pd.DataFrame({'Scores': self.train_losses,
                           'Cycle': self.all_cycles,
                           'Epoch': self.all_epochs,
                           'Index': np.arange(0, len(self.train_losses))})

        # Plot cycle borders
        np_cycles = np.unique(np.array(self.all_cycles))
        for cycle in np_cycles:
            local_df = df[df['Cycle'] == cycle]

            plt.plot([min(local_df['Index']),min(local_df['Index'])],
                     [min(df['Scores']), max(df['Scores'])],
                     c='black', alpha=0.5)

        plt.plot(df['Index'], df['Scores'], '-ok', c='blue', alpha=0.8)
        plt.ylabel('Train loss', fontsize=15)
        plt.xlabel('Step', fontsize=15)
        plt.grid()
        plt.show()

    def _save_nn(self, model_zip: str, model_pth: str) -> None:
        """
        The method saves the neural network to the specified folder

        :param model_zip: name of the file to save the model to
        :param model_pth: name of the file to save the optimizer to
        """

        # Determine input dimensions for data
        for index, param_tensor in enumerate(self.nn_model.state_dict()):
            if index == 0:
                x = self.nn_model.state_dict()[param_tensor].size()
                x = list(x)
                example_inputs = torch.ones(x)
            else:
                pass

        # Create TorchScript by tracing the computation graph with an example input
        net_trace = jit.trace(self.nn_model, example_inputs)
        actual_model_path = os.path.join(self.logs_path, model_zip)
        jit.save(net_trace, actual_model_path)

        # Save optimizer configuration to file
        actual_opt_path = os.path.join(self.logs_path, model_pth)
        torch.save({'optimizer': self.nn_optimizer.state_dict()}, actual_opt_path)

        # Save state of actual path to model
        self.actual_model_path = actual_model_path
        self.actual_opt_path = actual_opt_path

    def get_actual_opt_path(self) -> str:
        """
        The method returns the path to the current version of the optimizer
        """
        return(self.actual_opt_path)

    def get_actual_model_path(self) -> str:
        """
        The method returns the path to the current version of the neural network
        """
        return(self.actual_model_path)

    @staticmethod
    def delete_nn(model_to_remove):
        # TODO Add ability to remove models from logs
        raise NotImplementedError("This functionality not implemented yet")


class PopulationLogger:
    """
    A class to support the training process of a population of neural networks

    :param nn_type: neural network architecture for optimization
    (available types 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param logs_path: path to the folder where saved versions of the neural network are stored
    :param pop_size: size of generated population
    :param cycles: number of cycles to optimize
    :param epoch_per_cycle: how many epochs should the neural network be trained
    after the crossover in each cycle
    """

    def __init__(self, logs_path: str,  nn_type: str, pop_size: int, cycles: int,
                 epoch_per_cycle: int):
        self.logs_path = os.path.join(logs_path, 'population')
        # Create folder if it doesnt exists
        if os.path.isdir(self.logs_path) == False:
            os.makedirs(self.logs_path)

        self.nn_type = nn_type
        self.pop_size = pop_size
        self.cycles = cycles
        self.epoch_per_cycle = epoch_per_cycle

        # Dictionary for saving metadata
        # With pre-defined structure
        self.pop_metadata = {}

        # Cycles start from 1, 2, 3, ...
        for cycle in range(1, self.cycles+1):
            self.pop_metadata.update({cycle:{}})

            # Models start from 0, 1, 2 ,3, ...
            for model_number in range(0, self.pop_size):
                self.pop_metadata.get(cycle).update(
                    {model_number: [np.zeros(self.epoch_per_cycle), '']})


    def collect_scores(self, model_number: int, current_cycle: int,
                       current_epoch: int, model_score: float, change: str) -> None:
        """
        Method for updating metadata during training of multiple neural networks
        (from one population)

        :param model_number: index of model (neural network) in population
        :param current_cycle: current cycle of the evolution process
        :param current_epoch: current epoch of the learning process
        :param model_score: value of the metric in the training sample
        :param change: descriptions of changes during mutation
        """

        # Get array with scores
        current_cycle_dict = self.pop_metadata.get(current_cycle)
        current_model_list = current_cycle_dict.get(model_number)
        current_model_scores = current_model_list[0]

        # Store train loss
        current_model_scores[current_epoch-1] = model_score

        # Update metadata
        current_model_list[0] = current_model_scores
        current_model_list[1] = change
        current_cycle_dict.update({model_number: current_model_list})
        self.pop_metadata.update({current_cycle:current_cycle_dict})

    def get_metadata(self, cycle: int) -> dict:
        """
        The method allow get metadata for a specific training cycle

        :param cycle: current cycle of the evolution process

        :return : dictionary with metadata
        """

        current_cycle_dict = self.pop_metadata.get(cycle)
        print(self.pop_metadata)
        return current_cycle_dict

    def save_metadata(self, where_to_save: str) -> None:
        """
        This method allow save metadata in a specified folder

        :param where_to_save: path to folder where "Metadata.json" file must be
        saved
        """

        # Create folder if it doesnt exists
        if os.path.isdir(where_to_save) == False:
            os.makedirs(where_to_save)

        # Save file as json
        json_path = os.path.join(where_to_save, 'Metadata.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.metadata, json_file)

    def save_nn(self, current_cycle: int, model_number: int, nn_model,
                nn_optimizer, nn_loss, nn_batch) -> None:
        """
        The method saves the neural network (as zip and pth files) in a special folder

        :param current_cycle: current cycle of the evolution process
        :param model_number: index of model (neural network) in population
        :param nn_model: NN model to save
        :param nn_optimizer: Optimizer to save
        :param nn_loss: Loss of NN model
        :param nn_batch: batch size
        """

        # Define archive files names
        model_zip = ''.join((str(current_cycle),'_',str(model_number),'_model.zip'))
        model_pth = ''.join((str(current_cycle),'_',str(model_number),'_optimizer.pth'))

        # Determine input dimensions for data
        for index, param_tensor in enumerate(nn_model.state_dict()):
            if index == 0:
                x = nn_model.state_dict()[param_tensor].size()
                x = list(x)
                example_inputs = torch.ones(x)
            else:
                pass

        # Create TorchScript by tracing the computation graph with an example input
        net_trace = jit.trace(nn_model, example_inputs)
        actual_model_path = os.path.join(self.logs_path, model_zip)
        jit.save(net_trace, actual_model_path)

        # Save optimizer configuration to file
        actual_opt_path = os.path.join(self.logs_path, model_pth)
        torch.save({'optimizer': nn_optimizer.state_dict()}, actual_opt_path)


def get_device():
    """
    Method for getting available devices.

    """

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device