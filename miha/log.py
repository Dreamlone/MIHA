import os
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from torch import jit
import torch.utils.data as data_utils
import pandas as pd
import numpy as np


class ModelLogger:
    """
    A class that stores the history of neural network training in the process
    of evolution and perform operations to save models, output messages with logs

    :param logs_path: path to the folder where saved versions of the neural network are stored
    """

    def __init__(self, logs_path: str):
        self.logs_path = logs_path

        self.hall_of_fame = {}
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
        self.nn_model.to('cpu')
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
                self.save_nn(model_zip, model_pth)
            else:
                pass
        # If there is cycle optimization started
        else:
            if is_last_epoch == True:
                model_zip = ''.join(('model_', str(current_cycle), '.zip'))
                model_pth = ''.join(('optimizer_', str(current_cycle), '.pth'))
                model_path = [os.path.join(self.logs_path, model_zip),
                              os.path.join(self.logs_path, model_pth)]

                # Save NN model
                self.save_nn(model_zip, model_pth)
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

        plt.plot(df['Index'], df['Scores'], '-ok', c='blue', alpha=0.8)
        plt.ylabel('Train loss', fontsize=15)
        plt.xlabel('Step', fontsize=15)
        plt.grid()
        plt.show()

    def save_nn(self, model_zip: str, model_pth: str) -> None:
        """
        The method saves the neural network to the specified folder

        :param model_zip: name of the file to save the model to
        :param model_pth: name of the file to save the optimizer to
        """

        # Create TorchScript by tracing the computation graph with an example input
        # TODO make x adaptive
        x = torch.ones(16, 1, 2, 2)
        net_trace = jit.trace(self.nn_model, x)
        actual_model_path = os.path.join(self.logs_path, model_zip)
        jit.save(net_trace, actual_model_path)

        # Save optimizer configuration to file
        actual_opt_path = os.path.join(self.logs_path, model_pth)
        torch.save({'optimizer': self.nn_optimizer.state_dict()}, actual_opt_path)

        # Save state of actual path to model
        self.actual_model_path = actual_model_path
        self.actual_opt_path = actual_opt_path

    def get_actual_opt_path(self) -> str:
        return(self.actual_opt_path)

    def get_actual_model_path(self) -> str:
        return(self.actual_model_path)

    @staticmethod
    def delete_nn(model_to_remove):
        # TODO Add ability to remove models from logs
        pass


