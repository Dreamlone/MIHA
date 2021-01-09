import torch
import torch.nn as nn

from miha.model import NNOptimizer

# Necessary for reproducibility of results
random_seed = 100
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Initialization CNN Autoencoder for remote sensing data gapfilling
# You need to prepare a class that will have "forward" function in addition to "initialization"
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=0.99, mode='nearest'))
        self.layer4 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=2, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=1.97, mode='bilinear'))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

# Passing the prepared architecture to the optimizer class
gapfilling_optimizer = NNOptimizer(nn_type = 'AE',
                                   task='regression',
                                   input = './data/remote_sensing_gapfilling/X_train.pt',
                                   output = './data/remote_sensing_gapfilling/Y_train.pt',
                                   cycles = 2,
                                   population_size = 2,
                                   epoch_per_cycle = 4,
                                   fixing_epochs = 4,
                                   save_logs = True,
                                   logs_folder = './data/remote_sensing_gapfilling/logs')

# Find the best model hyperparameters set for chosen topology
best_solution = gapfilling_optimizer.optimize(source_nn = ConvNet,
                                              source_loss = nn.MSELoss,
                                              source_optimizer = torch.optim.Adam,
                                              source_batch_size = 32)