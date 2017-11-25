import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_features, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_classes)

    def forward(self, x, training=True):
        # a neural network with 2 hidden layers
        # x -> FC -> relu -> dropout -> FC -> relu -> dropout -> FC -> output
        x = F.relu(self.layer1(x))
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, 0.5, training=training)
        x = self.layer3(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x, training=False))
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(),
                                       y.float()).float()) / y.size()[0]
        return acc.cpu().data[0]
