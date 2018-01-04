import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import utils

"""
A simple model based loosely off of Andrej Karpathy's model in his reinforcement learning blog post.
"""

class Model(nn.Module):
    def __init__(self, input_dim, action_dim, batch_norm=False):
        super(Model, self).__init__()
        self.obs_space = input_dim
        self.act_space = action_dim
        self.bnorm = batch_norm
        self.hidden_dim = action_dim*100

        self.entry = nn.Linear(input_dim, self.hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm1d(self.hidden_dim)

        self.action_out = nn.Linear(self.hidden_dim, action_dim)
        self.value_out = nn.Linear(self.hidden_dim, 1)

        self.dropout = nn.Dropout(.2)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, x, requires_value=True, dropout=False):
        fx = F.relu(self.entry(x))
        if self.bnorm: fx = self.bnorm1(fx)
        if dropout: fx = self.dropout(fx)
        fx = F.relu(self.hidden(fx))
        if self.bnorm: fx = self.bnorm2(fx)
        action = self.action_out(fx)
        if requires_value:
            value = self.value_out(fx)
            return value, action
        else:
            return 0, action

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking in in pytorch...
        """
        for param in list(self.parameters()):
            if torch.sum(param.data != param.data) > 0:
                print(param)
