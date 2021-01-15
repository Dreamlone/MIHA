import torch
import torch.nn as nn

from miha.model import NNOptimizer

# Necessary for reproducibility of results
random_seed = 100
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Initialization simple FNN for MNIST classification task
# You need to prepare a class that will have "forward" function in addition to "initialization"
class FnnClassifier(nn.Module):

    def __init__(self):
        super(FnnClassifier, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, 500),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(500, 10),
                                    nn.LogSoftmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

# Passing the prepared architecture to the optimizer class
gapfilling_optimizer = NNOptimizer(nn_type = 'FNN',
                                   task='classification',
                                   input='./data/mnist/X_train.pt',
                                   output='./data/mnist/Y_train.pt',
                                   cycles=6,
                                   population_size=4,
                                   epoch_per_cycle=4,
                                   fixing_epochs=10,
                                   runup_epochs=10,
                                   save_logs=True,
                                   logs_folder='D:/miha_exp/mnist/2')
# './data/mnist/logs'
# Find the best model hyperparameters set for chosen topology
best_solution = gapfilling_optimizer.optimize(source_nn=FnnClassifier,
                                              source_loss=nn.functional.nll_loss,
                                              source_optimizer=torch.optim.Adam,
                                              source_batch_size=32,
                                              crossover=False,
                                              check_mode=False)