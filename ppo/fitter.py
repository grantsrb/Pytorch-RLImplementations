import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import utils
from utils import discount, gae
import model

if torch.cuda.is_available():
    torch.FloatTensor = torch.cuda.FloatTensor

class Fitter():
    def __init__(self, net, optimizer, pool, val_const=.5, entropy_const=0.01, spatio_const=0.001, gamma=0.99, lambda_=0.97, predict_spatio=False):
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

        #self.old_datas = [[]]*n_olddatas
        #self.n_olddatas = n_olddatas
        #self.data_idx = 0


    def fit(self, new_data, epochs=10, batch_size=128, clip_const=0.2, max_norm=0.5):
        """
        Runs stochastic gradient descent on the collected data.

        new_data: a sequence of data lists in the following order
            actions, observations, discounted rewards, pi values, advantages
        """

        """
        Each array of data comes in with a shape of [step, env]. To get each
        env's step to line up in a single long array, we need to transpose each array
        and then ravel them.
        """
        actions, observs, rewards, values, deltas, dones = new_data

        async_arr = [self.pool.apply_async(discount, [rewards[:,i], self.gamma, dones[:,i]]) for i in range(rewards.shape[1])]
        rewards = [async.get() for async in async_arr]
        rewards = Variable(torch.FloatTensor(rewards).view(-1))
        actions = torch.LongTensor(actions).permute(1,0).contiguous().view(-1)
        observs = observs.swapaxes(0,1).reshape((-1,)+self.obs_space)
        observs = Variable(torch.from_numpy(observs).float())

        old_vals, old_pis, spatios, spatio_preds = self.net.forward(observs)
        old_vals = Variable(old_vals.data)
        old_pis = old_pis.data[torch.arange(0,old_pis.size(0)).long(),actions]

        async_arr = [self.pool.apply_async(discount, [deltas[:,i], self.gamma*self.lambda_, dones[:,i]]) for i in range(deltas.shape[1])]
        advantages = [async.get() for async in async_arr]
        advantages = Variable(torch.FloatTensor(advantages).view(-1))
        returns = Variable(advantages.data + old_vals.data.squeeze())
        #advantages = Variable((advantages-torch.mean(advantages))/(torch.std(advantages)+1e-7))

        old_pis = torch.clamp(old_pis, 1e-7, 1)
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
                self.optimizer.zero_grad()
                idxs = indices[i*batch_size:(i+1)*batch_size]
                batch_acts = actions[idxs]
                batch_obs = observs[idxs]
                batch_advs = advantages[idxs]
                batch_returns = returns[idxs]
                batch_old_pis = old_pis[idxs]
                batch_old_vals = old_vals[idxs]
                batch_rewards = rewards[idxs]

                values, pi_raw, spatios, spatio_preds = self.net.forward(batch_obs)

                probs = self.net.softmax(pi_raw)
                new_pis = probs[torch.arange(0,probs.size()[0]).long(), batch_acts]
                ratios = new_pis/batch_old_pis
                clipped_ratios = torch.clamp(ratios, 1.0-clip_const, 1.0+clip_const)
                clip_loss = -torch.mean(torch.min(ratios*batch_advs, clipped_ratios*batch_advs))

                vals_clipped = batch_old_vals + torch.clamp(values.squeeze()-batch_old_vals, -clip_const, clip_const)
                v1 = (values.squeeze() - batch_rewards)**2
                v2 = (vals_clipped - batch_rewards)**2
                val_loss = self.val_const * .5 * torch.mean(torch.max(v1, v2))

                logprobs = self.net.logsoftmax(pi_raw)
                entropy_loss = -self.entropy_const * torch.mean(probs*logprobs)

                loss = clip_loss + val_loss - entropy_loss

                running_loss += loss.data[0]
                running_clip += clip_loss.data[0]
                running_val += val_loss.data[0]
                running_entropy += entropy_loss.data[0]

                loss.backward()
                norm = nn.utils.clip_grad_norm(self.net.parameters(), max_norm)
                self.optimizer.step()

            self.net.check_grads()
            avg_loss += running_loss/n_data_pts
            avg_clip += running_clip/n_data_pts
            avg_val += running_val/n_data_pts
            avg_entropy += running_entropy/n_data_pts

        return avg_loss/epochs, avg_clip/epochs, avg_val/epochs, avg_entropy/epochs, 0

        # Update data
        # self.old_datas[self.data_idx] = new_data
        # self.data_idx = (self.data_idx + 1) % self.n_olddatas
