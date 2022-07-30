import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.act = F.leaky_relu

        # model is severely underfitting right now,
        # so better to turn down the regularization,
        # by making the dropout probability low
        self.drop = nn.Dropout(0.025)

    def forward(self, x):
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x
