import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F
import math


class Model(nn.Module):
    def __init__(self, obs_space, act_space, batch_norm=False):
        super(Model, self).__init__()

        self.obs_space = obs_space
        self.act_space = act_space
        self.batch_norm = batch_norm

        self.convs = nn.ModuleList([])
        self.bnorms = nn.ModuleList([])

        self.convs.append(nn.Conv2d(obs_space[-3], 24, 9, stride=3))
        self.bnorms.append(nn.BatchNorm2d(24))
        height1 = math.ceil((obs_space[-2] - 9 + 1)/float(3))
        width1 = math.ceil((obs_space[-1] - 9 + 1)/float(3))


        self.convs.append(nn.Conv2d(24, 48, 5, stride=2))
        self.bnorms.append(nn.BatchNorm2d(48))
        height2 = math.ceil((height1 - 5 + 1)/float(2))
        width2 = math.ceil((width1 - 5 + 1)/float(2))

        self.convs.append(nn.Conv2d(48, 64, 3, stride=1))
        self.bnorms.append(nn.BatchNorm2d(64))
        height3 = math.ceil((height2 - 3 + 1)/float(1))
        width3 = math.ceil((width2 - 3 + 1)/float(1))

        self.flat_size = height3*width3*64

        self.img_pred1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.imgnorm1 = nn.BatchNorm2d(64)
        self.img_pred2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(self.flat_size, 200)
        self.fcnorm1 = nn.BatchNorm1d(200)
        self.q_out = nn.Linear(200, act_space)
        self.val = nn.Linear(200, 1)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.mseloss = nn.MSELoss()


    def forward(self, x):
        fx = x

        for conv,bnorm in zip(self.convs,self.bnorms):
            fx = conv(fx)
            fx = self.relu(fx)
            fx = bnorm(fx) if self.batch_norm else fx

        concept = fx.view(-1,self.flat_size)
        conc_pred = self.img_pred1(fx)
        conc_pred = self.relu(conc_pred)
        conc_pred = self.imgnorm1(conc_pred) if self.batch_norm else conc_pred
        conc_pred = self.img_pred2(conc_pred)
        conc_pred = self.relu(conc_pred)

        conc_pred = conc_pred.view(-1,self.flat_size)
        fx = conc_pred
        fx = self.fc1(fx)
        fx = self.fcnorm1(fx) if self.batch_norm else fx
        fx = self.relu(fx)

        q = self.q_out(fx)
        val = self.val(fx)

        return val, q, concept, conc_pred

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking in in pytorch...
        """
        for param in list(self.parameters()):
            if torch.sum(param.data != param.data) > 0:
                print(param)
