import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import utils
from utils import discount, gae
import model

class Fitter():
    def __init__(self, input_dim, action_dim, n_olddatas=3):

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.mseloss = nn.MSELoss()

        self.old_datas = [[]]*n_olddatas
        self.n_olddatas = n_olddatas
        self.data_idx = 0


    def fit_policy(self, net, new_data, optimizer, epochs=10, batch_size=128, clip_const=0.2, max_norm=0.5, val_const=.5, entropy_const=0.01, gamma=0.99, lambda_=0.97):
        """
        Runs stochastic gradient descent on the collected data.

        new_data: a sequence of data lists in the following order
            actions, observations, discounted rewards, pi values, advantages
        """

        # old_net = model.Model(self.input_dim, self.action_dim)
        # old_net.load_state_dict(net.state_dict())
        # for p in old_net.parameters():
        #     p.trainable = False

        if self.old_datas[0] == []:
            data = new_data
        else:
            data = [*new_data]
            for od in self.old_datas:
                for i,d in enumerate(od):
                    assert type(d) == type([])
                    assert type(data[i]) == type([])
                    data[i] = data[i] + d

        actions, observs, rewards, advantages, old_pis, old_vals, mask = data

        rewards = discount(rewards, gamma, mask)
        rewards = Variable(torch.FloatTensor(rewards))
        actions = torch.LongTensor(actions)
        observs = Variable(torch.from_numpy(np.asarray(observs)).float())

        old_vals = Variable(torch.FloatTensor(old_vals))
        advantages = Variable(torch.FloatTensor(advantages))
        returns = Variable(advantages.data + old_vals.data)
        #advantages = Variable((advantages-torch.mean(advantages))/(torch.std(advantages)+1e-7))

        old_pis = torch.clamp(torch.FloatTensor(old_pis), 1e-7, 1)
        old_pis = Variable(old_pis)

        n_data_pts = len(actions)
        n_loops = n_data_pts//batch_size
        avg_loss = 0
        avg_clip = 0
        avg_val = 0
        avg_entropy = 0
        for epoch in range(epochs):
            indices = torch.randperm(n_data_pts)
            running_loss,running_clip,running_val,running_entropy = 0,0,0,0

            for i in range(n_loops):
                optimizer.zero_grad()
                idxs = indices[i*batch_size:(i+1)*batch_size]
                batch_acts = actions[idxs]
                batch_obs = observs[idxs]
                batch_advs = advantages[idxs]
                batch_returns = returns[idxs]
                batch_old_pis = old_pis[idxs]
                batch_old_vals = old_vals[idxs]
                batch_rewards = rewards[idxs]

                values, pi_raw = net.forward(batch_obs)

                probs = self.softmax(pi_raw)
                new_pis = probs[torch.arange(0,probs.size()[0]).long(), batch_acts]
                ratios = new_pis/batch_old_pis
                clipped_ratios = torch.clamp(ratios, 1.0-clip_const, 1.0+clip_const)
                clip_loss = -torch.mean(torch.min(ratios*batch_advs, clipped_ratios*batch_advs))

                vals_clipped = batch_old_vals + torch.clamp(values.squeeze()-batch_old_vals, -clip_const, clip_const)
                v1 = (values.squeeze() - batch_returns)**2
                v2 = (vals_clipped - batch_returns)**2
                val_loss = val_const * .5 * torch.mean(torch.max(v1, v2))

                logprobs = self.logsoftmax(pi_raw)
                entropy_loss = -entropy_const * torch.mean(probs*logprobs)

                loss = clip_loss + val_loss - entropy_loss

                running_loss += loss.data[0]
                running_clip += clip_loss.data[0]
                running_val += val_loss.data[0]
                running_entropy += entropy_loss.data[0]

                loss.backward()
                norm = nn.utils.clip_grad_norm(net.parameters(), max_norm)
                optimizer.step()

            net.check_grads()
            avg_loss += running_loss/n_data_pts
            avg_clip += running_clip/n_data_pts
            avg_val += running_val/n_data_pts
            avg_entropy += running_entropy/n_data_pts

        print("Loss",avg_loss/epochs,"– Clip:",avg_clip/epochs,"– Val:",avg_val/epochs,"– Entr:",avg_entropy/epochs)

        # Update data
        # self.old_datas[self.data_idx] = new_data
        # self.data_idx = (self.data_idx + 1) % self.n_olddatas
