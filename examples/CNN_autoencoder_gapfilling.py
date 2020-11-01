import torch.nn as nn
from miha.optimizer import CnnAutoencoder

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
        self.drop_out = nn.Dropout(0.1)
        self.layer4 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=2, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=1.97, mode='bilinear'))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.drop_out(out)
        out = self.layer4(out)
        return (out)

# Passing the prepared architecture to the optimizer class
gapfilling_optimizer = CnnAutoencoder()

# Find the best model hyperparameters set for chosen topology
best_solution = gapfilling_optimizer.optimize(ConvNet)