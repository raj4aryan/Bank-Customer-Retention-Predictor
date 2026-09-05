import torch.nn as nn

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# class RegressionModel(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.layer1 = nn.Linear(input_size, 64)
#         self.relu1 = nn.ReLU()
#         self.layer2 = nn.Linear(64, 32)
#         self.relu2 = nn.ReLU()
#         self.layer3 = nn.Linear(32, 16)
#         self.relu3 = nn.ReLU()
#         self.outputLayer = nn.Linear(16, 1)

#     def forward(self, x):
#         x = self.relu1(self.layer1(x))
#         x = self.relu2(self.layer2(x))
#         x = self.relu3(self.layer3(x))
#         x = self.outputLayer(x)
#         return x
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # We define the entire sequence of machines in one block
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Raw output for regression
        )

    def forward(self, x):
        # The assembly line is automated. We just hand the data to the network.
        return self.network(x)

