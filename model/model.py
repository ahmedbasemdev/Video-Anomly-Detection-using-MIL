import torch.nn as nn


class AnomalyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, my_input):
        net = self.fc1(my_input)
        net = self.relu(net)
        net = self.dropout(net)

        net = self.relu(self.fc2(net))
        net = self.dropout(net)

        output = self.sigmoid(self.fc3(net))

        return output
