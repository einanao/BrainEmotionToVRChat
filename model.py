import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 5)
        self.fc3 = nn.Linear(5, 2)
        self.act = F.leaky_relu
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        x = self.act(self.fc3(x))
        return x
