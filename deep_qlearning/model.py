import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

"""
Description: A simple model based loosely off of Andrej Karpathy's model in his reinforcement learning blog post.
"""

class Model(nn.Module):
    def __init__(self, obs_space, act_space, batch_norm=False):
        super(Model, self).__init__()

        self.obs_space = obs_space
        self.act_space = act_space
        self.batch_norm = batch_norm

        self.hidden_dim = act_space*100

        self.flat_size = obs_space[-1]*obs_space[-2]*obs_space[-3]

        self.entry = nn.Linear(self.flat_size, self.hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm1d(self.hidden_dim)

        self.q_out = nn.Linear(self.hidden_dim, act_space)
        self.value_out = nn.Linear(self.hidden_dim, 1)

        self.dropout = nn.Dropout(0)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.mseloss = nn.MSELoss()

    def forward(self, x):
        fx = self.relu(self.entry(x.view(x.size(0), -1)))
        if self.batch_norm: fx = self.bnorm1(fx)
        fx = self.relu(self.hidden(fx))
        if self.batch_norm: fx = self.bnorm2(fx)
        q = self.q_out(fx)
        value = self.value_out(fx)
        return value, q, 0, 0 # Zeros are necessary for code structure

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking in in pytorch...
        """
        for param in list(self.parameters()):
            if torch.sum(param.data != param.data) > 0:
                print(param)
