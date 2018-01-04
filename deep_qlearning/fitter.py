import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import utils
from utils import discount, gae
import model
import copy

if torch.cuda.is_available():
    torch.FloatTensor = torch.cuda.FloatTensor

class Fitter():
    def __init__(self, net, optimizer, pool, val_const=.5, entropy_const=0.01, spatio_const=0.001, gamma=0.99, lambda_=0.97, predict_spatio=False, max_observs=10000):
        self.obs_space = net.obs_space
        self.act_space = net.act_space
        self.net = net
        self.optimizer = optimizer
        self.pool = pool
        self.val_const = val_const
        self.entropy_const = entropy_const
        self.gamma = gamma
        self.lambda_ = lambda_
        self.spatio_const = spatio_const
        self.predict_spatio = predict_spatio

        self.max_observs = max_observs
        self.old_observs = []
        self.old_rewards = []
        self.old_dones = []
        self.old_actions = []


    def fit(self, new_data, epochs=10, batch_size=128, max_norm=0.5):
        """
        Runs stochastic gradient descent on the collected data.

        new_data: a sequence of data lists in the following order
            actions, observations, discounted rewards, q values, advantages
        """

        """
        Each array of data comes in with a shape of [step, env]. To get each
        env's step to line up in a single long array, we need to transpose each array
        and then ravel them.
        """

        actions, observs, rewards, values, deltas, dones = new_data

        async_arr = [self.pool.apply_async(discount, [rewards[:,i], self.gamma, dones[:,i]]) for i in range(rewards.shape[1])]
        rewards = [async.get() for async in async_arr]
        rewards = torch.FloatTensor(rewards).view(-1)
        actions = torch.LongTensor(actions).permute(1,0).contiguous().view(-1)
        observs = observs.swapaxes(0,1).reshape((-1,)+self.obs_space)
        observs = torch.from_numpy(observs).float()
        dones = 1.-dones.swapaxes(0,1).reshape((-1,))
        dones = torch.from_numpy(dones).float()
        if torch.cuda.is_available():
            observs = observs.cuda()
            dones = dones.cuda()

        if type(self.old_observs) != type([]):
            self.old_observs = torch.cat([self.old_observs, observs], 0)
            self.old_rewards = torch.cat([self.old_rewards, rewards], 0)
            self.old_dones = torch.cat([self.old_dones, dones], 0)
            self.old_actions = torch.cat([self.old_actions, actions], 0)
        else:
            self.old_observs, self.old_rewards, self.old_dones, self.old_actions = observs, rewards, dones, actions
        if self.old_observs.size(0) > self.max_observs:
            self.old_observs = self.old_observs[observs.size(0):]
            self.old_rewards = self.old_rewards[rewards.size(0):]
            self.old_dones = self.old_dones[dones.size(0):]
            self.old_actions = self.old_actions[actions.size(0):]
        observs = Variable(self.old_observs)
        rewards = Variable(self.old_rewards)
        dones = self.old_dones
        actions = self.old_actions

        self.net.train(mode=False)
        old_vals, old_qs, spatios, spatio_preds = self.net.forward(observs)
        old_vals = Variable(old_vals.data)
        old_qs = Variable(old_qs.data[torch.arange(0,old_qs.size(0)).long(),actions])
        self.net.train(mode=True)

        q_targets = Variable(rewards.data[:-1] + self.gamma*old_qs.data[1:]*dones[:-1])

        n_data_pts = len(actions)
        n_loops = (n_data_pts-1)//batch_size
        avg_loss = 0
        for epoch in range(epochs):
            indices = torch.randperm(n_data_pts-1)
            running_loss = 0

            for i in range(n_loops):
                self.optimizer.zero_grad()
                idxs = indices[i*batch_size:(i+1)*batch_size]
                batch_acts = actions[idxs]
                batch_obs = observs[idxs]
                batch_targets = q_targets[idxs]

                values, q_vals, spatios, spatio_preds = self.net.forward(batch_obs)
                new_qs = q_vals[torch.arange(0,q_vals.size()[0]).long(), batch_acts]

                loss = self.net.mseloss(new_qs, batch_targets)
                running_loss += loss.data[0]

                loss.backward()
                norm = nn.utils.clip_grad_norm(self.net.parameters(), max_norm)
                self.optimizer.step()

            self.net.check_grads()
            avg_loss += running_loss/n_data_pts

        return avg_loss/epochs

        # Update data
        # self.old_datas[self.data_idx] = new_data
        # self.data_idx = (self.data_idx + 1) % self.n_olddatas
