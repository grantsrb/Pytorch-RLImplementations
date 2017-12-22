import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import utils
from multiprocessing import Pool
from utils import discount

class A2C():
    """
    Class to create the loss functions for A2C.
    """

    def __init__(self, net, n_envs, pool, val_const=0.5, entropy_const=0.01, spatio_const=0.001, gamma=0.99, lambda_=0.97, predict_spatio=False):
        self.net = net
        self.n_envs = n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.predict_spatio = predict_spatio
        self.pool = pool
        self.val_const = val_const
        self.entropy_const = entropy_const
        self.spatio_const = spatio_const

    def step(self, ep_actions, ep_observs, ep_rewards, ep_values, ep_deltas, ep_dones):

        # Discount rewards
        async_arr = [self.pool.apply_async(discount, [ep_rewards[:,i], self.gamma, ep_dones[:,i]]) for i in range(self.n_envs)]
        rewards = [async.get() for async in async_arr]
        t_rewards = Variable(torch.FloatTensor(rewards).view(-1))

        # Evaluate pis and values
        ep_observs = ep_observs.swapaxes(0,1).reshape((-1,)+self.net.obs_space)
        if torch.cuda.is_available():
            t_observs = Variable(torch.from_numpy(ep_observs).cuda())
        else:
            t_observs = Variable(torch.from_numpy(ep_observs))
        values, raw_actions, spatios, spatio_preds = self.net.forward(t_observs)
        spatios = Variable(spatios.data) # Detach graph
        softlogs = self.net.logsoftmax(raw_actions)
        actions = torch.LongTensor(ep_actions).permute(1,0).contiguous().view(-1)
        if torch.cuda.is_available():
            action_logs = softlogs[torch.arange(0,len(actions)).long().cuda(),actions] # Only want action predictions for the performed ep_actions
        else:
            action_logs = softlogs[torch.arange(0,len(actions)).long(),actions]

        # Create advantages from deltas (GAE)
        async_arr = [self.pool.apply_async(discount, [ep_deltas[:,i], self.gamma*self.lambda_, ep_dones[:,i]]) for i in range(self.n_envs)]
        advantages = [async.get() for async in async_arr]
        advantages = Variable(torch.FloatTensor(advantages).view(-1))
        #returns = Variable(advantages.data + values.data.squeeze())

        # Evaluate Losses
        action_loss = -torch.mean(action_logs*advantages) # Standard policy gradient
        value_loss = self.val_const*self.net.mseloss(values.squeeze(), t_rewards)
        entropy = -self.entropy_const*torch.mean(self.net.softmax(raw_actions)*softlogs)
        if self.predict_spatio:
            mask = torch.from_numpy(1.-ep_dones).float()
            if torch.cuda.is_available():
                mask = mask.cuda()
            mask = Variable(mask.permute(1,0).contiguous().view(-1))
            spatio_loss = self.spatio_const*torch.mean(torch.sum((spatio_preds[:-1]-spatios[1:])**2, dim=-1)*mask[:-1])
        else:
            spatio_loss = 0

        loss = action_loss + value_loss - entropy + spatio_loss
        loss.backward()
        self.net.check_grads() # Checks for NaN in gradients. Pytorch can be finicky

        return loss, action_loss, value_loss, entropy, spatio_loss
