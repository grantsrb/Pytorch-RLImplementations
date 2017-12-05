import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gc
import resource
import sys
import model



# hyperparameters
gamma = .99
lambda_ = .96
batch_size = 10
lr = 1e-3/batch_size # Divide by batchsize as a shortcut to averaging gradients over batches
action_dim = 2
net_save_file = "net_state_dict.p"
optim_save_file = "optim_state_dict.p"

resume = False
render = False
if len(sys.argv) > 1:
    resume = bool(sys.argv[1])
    if len(sys.argv) > 2:
        render = bool(sys.argv[2])


# Make preprocessing function
def preprocess(pic):
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic.ravel()

# Make discounting function
def discount(rs, disc_factor):
    discounteds = [0]*len(rs)
    running_sum = 0
    for i in reversed(range(len(rs))):
        if rs[i] != 0: running_sum = 0
        running_sum = running_sum*disc_factor + rs[i]
        discounteds[i] = running_sum
    return discounteds

def calc_temporal_diffs(rewards, values, gamma):
    assert type(rewards) == type(values)
    return rewards + gamma*values[1:] - values[:-1]

def sum_one(action_vec):
    new_vec = [0]*len(action_vec)
    running_sum = 0
    for i in range(len(action_vec)-1):
        new_vec[i] = round(action_vec[i], 4)
        running_sum += new_vec[i]
    new_vec[-1] = 1-running_sum
    return new_vec
    
# Make environment
env = gym.make("Pong-v0")
observation = env.reset()

# Make model and optimizer
prepped_state = preprocess(observation)
net = model.Model(prepped_state.shape[0], action_dim)
optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer.zero_grad()
if resume:
    net.load_state_dict(torch.load(net_save_file))  
    optimizer.load_state_dict(torch.load(optim_save_file))

logsoftmax = nn.LogSoftmax()
softmax = nn.Softmax()

# Store actions, states, values
actions, states, action_data, rewards = [], [], [], []
episode_reward = 0
baseline = 0
avg_reward = None
prev_obs = 0

net.train(mode=False)
episode, epoch = 0, 0

while True:
    if render: env.render()

    # Prep state
    observation = preprocess(observation)
    state = observation-prev_obs 
    prev_obs = observation
    states.append(state)

    # Take action
    state = torch.from_numpy(state).view(1,-1).float()
    t_state = Variable(state)
    value, action_pred = net.forward(t_state, requires_value=False)
    intmd = softmax(action_pred)
    pvec = sum_one(intmd.data.tolist()[0])
    action = np.random.choice(action_dim, p=pvec)
    action_data.append(action)
    one_hot = [0]*action_dim
    one_hot[action] = 1

    observation, reward, done, info = env.step(action+2) # Add two because possible actions are 2,3

    rewards.append(reward)
    actions.append(one_hot)
    episode_reward += reward

    if done:
        net.train(mode=True)
        episode += 1
        print("Finish Episode", episode, "–– Reward:", episode_reward, "–– Avg Action:", np.mean(action_data))

        rewards = discount(rewards, gamma)
        t_rewards = Variable(torch.FloatTensor(rewards))
        t_states = Variable(torch.from_numpy(np.asarray(states)).float())
        t_actions = Variable(torch.FloatTensor(actions))
        values, action_preds = net.forward(t_states, requires_value=False)
        action_preds = logsoftmax(action_preds)
        action_loss = -torch.sum(t_actions*action_preds, dim=1)
        advantages = (t_rewards - torch.mean(t_rewards))/torch.std(t_rewards)
        loss = torch.sum(action_loss*advantages)
        loss.backward()

#        tds = calc_temporal_diffs(rewards, values.squeeze(), gamma)
#        advantages = discount(tds, gamma*lambda_)
#        discounted_rewards = discount(rewards, gamma)
#        value_loss = torch.sqrt(torch.sum((values-discounted_rewards)**2))
#        loss = torch.mean(loss_step*advantages) + value_loss
        
        avg_reward = episode_reward if avg_reward == None else .99*avg_reward + .01*episode_reward

        if episode % batch_size == 0:
            epoch += 1
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch", epoch, "–– Updating Model –– Average Reward:", avg_reward)
            torch.save(net.state_dict(), net_save_file)
            torch.save(optimizer.state_dict(), optim_save_file)
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print("{:.2f} MB".format(max_mem_used / 1024))
        
        episode_reward = 0
        actions, states, action_data, rewards  = [], [], [], []
        observation = env.reset()
        prev_obs = 0
        net.train(mode=False)











