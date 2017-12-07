import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

"""
Description: A simple model based loosely off of Andrej Karpathy's model in his reinforcement learning blog post.
"""

class Model(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Model, self).__init__()
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
        self.mseloss = nn.MSELoss()

        self.old_data = None

    def forward(self, x, requires_value=True, dropout=False, bnorm=False):
        fx = F.relu(self.entry(x))
        if bnorm: fx = self.bnorm1(fx)
        if dropout: fx = self.dropout(fx)
        fx = F.relu(self.hidden(fx))
        if bnorm: fx = self.bnorm2(fx)
        action = self.action_out(fx)
        if requires_value:
            value = self.value_out(fx)
            return value, action
        else:
            return 0, action

    def fit_policy(self, new_data, optimizer, epochs=10, batch_size=128, clip_const=0.2, max_norm=0.5, val_const=.5, entropy_const=0.01):
        """
        Runs stochastic gradient descent on the collected data.

        new_data: a sequence of data lists in the following order
            actions, observations, discounted rewards, pi values, advantages
        """
        
        if self.old_data == None:
            data = new_data
        else:
            data = []
            for o,n in zip(self.old_data,new_data):
                assert type(o) == type([])
                assert type(n) == type([])
                data.append(o+n)

        actions, observs, rewards, old_pis, advantages = data

        actions = torch.LongTensor(actions)
        observs = Variable(torch.from_numpy(np.asarray(observs)).float())
        rewards = Variable(torch.FloatTensor(rewards))
        old_pis = Variable(torch.FloatTensor(old_pis))
        old_pis = torch.clamp(old_pis, 1e-4, 1)
        advantages = Variable(torch.FloatTensor(advantages))
        
        n_data_pts = len(actions)
        avg_loss = 0
        avg_clip = 0
        avg_val = 0
        avg_entropy = 0
        for epoch in range(epochs):
            indices = torch.randperm(n_data_pts)
            running_loss = 0
            running_clip = 0
            running_val = 0
            running_entropy = 0
            for i in range(0,n_data_pts,batch_size):
                optimizer.zero_grad()
                idxs = indices[i:i+batch_size]
                batch_acts = actions[idxs]
                batch_obs = observs[idxs]
                batch_rs = rewards[idxs]
                batch_old_pis = old_pis[idxs]
                batch_advs = advantages[idxs]
                
                values, raw_actions = self.forward(batch_obs)
                probs = self.softmax(raw_actions)
                new_pis = probs[torch.arange(0,probs.size()[0]).long(), batch_acts]
                ratios = new_pis/batch_old_pis
                clipped_ratios = torch.clamp(ratios, 1-clip_const, 1+clip_const)
                clip_loss = -torch.mean(torch.min(ratios*batch_advs, clipped_ratios*batch_advs))

                val_loss = val_const*self.mseloss(values.squeeze(), batch_rs)

                logprobs = self.logsoftmax(raw_actions)
                entropy_loss = -entropy_const*torch.mean(probs*logprobs)

                loss = clip_loss + val_loss - entropy_loss
                running_loss += loss.data[0]
                running_clip += clip_loss.data[0]
                running_val += val_loss.data[0]
                running_entropy += entropy_loss.data[0]

                loss.backward()
                norm = nn.utils.clip_grad_norm(self.parameters(), max_norm)
                optimizer.step()

            avg_loss += running_loss/n_data_pts
            avg_clip += running_clip/n_data_pts
            avg_val += running_val/n_data_pts
            avg_entropy += running_entropy/n_data_pts

        print("Loss",avg_loss/epochs,"– Clip:",avg_clip/epochs,"– Val:",avg_val/epochs,"– Entr:",avg_entropy/epochs)
        
        # Update old_pis
        new_actions, new_observs, _, _, _ = new_data
        actions = torch.LongTensor(new_actions)
        observs = Variable(torch.from_numpy(np.asarray(new_observs)).float())
        values, raw_outputs = self.forward(observs)
        probs = self.softmax(raw_outputs)
        old_pis = probs[torch.arange(0,probs.size()[0]).long(), actions]
        self.old_data = [new_data[i] for i in range(len(new_data))]
        self.old_data[3] = old_pis.data.tolist()

                
    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking in in pytorch...
        """
        for param in list(self.parameters()):
            if torch.sum(param.data != param.data) > 0:
                print(param)

    def add_model_gradients(self, model2):
        """
        Adds each of model2's parameters' gradients to the corresponding gradients of this model
        """
        params = list(self.parameters())
        for i,p in enumerate(model2.parameters()):
            if type(p.grad) != type(None):
                if type(params[i].grad) != type(None):
                    params[i].grad = params[i].grad + p.grad
                else:
                    params[i].grad = p.grad
                    
    def swap_weights(self, file1, file2):
        """
        Saves current state dict to file1 and loads state dict from file2
        """
        assert os.path.isfile(file2) == True
        torch.save(self.state_dict(), file1)
        self.load_state_dict(torch.load(file2))









