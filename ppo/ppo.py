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
from utils import preprocess, discount, sum_one


# hyperparameters
gamma = .99 # Discount factor
lambda_ = .97 # GAE moving average factor
clip_const = 0.2
ep_batch_size = 2
fit_batch_size = 256
n_olddatas = 5
n_epochs = 5
n_envs = 20 # Number of environments to operate in parallel (note that this implementation does not run the environments on seperate threads)
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
max_norm = 0.5 # Scales the gradients using their norm
max_tsteps = 80e6 # The number of environmental steps to take before ending the algorithm
lr = 1e-3/ep_batch_size/n_envs # Divide by batchsize as a shortcut to averaging the gradient over multiple batches

net_save_file = "net_state_dict.p"
optim_save_file = "optim_state_dict.p"

resume = False
render = False
if len(sys.argv) > 1:
    resume = bool(sys.argv[1])
    if len(sys.argv) > 2:
        render = bool(sys.argv[2])


# Make environments
envs = [gym.make("Pong-v0") for i in range(n_envs)]
for i,env in enumerate(envs):
    env.seed(i)
obs_bookmarks = [env.reset() for env in envs] # Used to track observations between environments
prev_bookmarks = [0 for i in range(n_envs)]

# Make model and optimizer
action_dim = 2 # Pong specific number of possible actions
prepped_state = preprocess(obs_bookmarks[0]) # Returns a vector representation of the observation
net = model.Model(prepped_state.shape[0], action_dim, n_olddatas=n_olddatas) 
optimizer = optim.Adam(net.parameters(), lr=lr)

if resume:
    net.load_state_dict(torch.load(net_save_file))
    optimizer.load_state_dict(torch.load(optim_save_file))

optimizer.zero_grad()

# Various functions that will be useful later
logsoftmax = nn.LogSoftmax()
softmax = nn.Softmax()
mseloss = nn.MSELoss()

# Store actions, observations, values
actions, observs, rewards, old_pis, advantages, mask = [], [], [], [], [], []
old_value = 0
episode_reward = 0
avg_reward = 0
running_reward = 0
reward_count = 0
avg_loss = None
episode, epoch = 0, 0
T = 0 # Tracks total environmental steps taken

net.train(mode=False)

while T < max_tsteps:

    for b in range(ep_batch_size):
        episode += 1
        for i,env in enumerate(envs):
            observation = obs_bookmarks[i]
            prev_obs = prev_bookmarks[i]
            
            for t in range(n_tsteps):
                T+=1

                if render and i == 0: env.render()

                # Prep observation
                observation = preprocess(observation)
                prepped_obs = observation-prev_obs # Gives some information about previous state
                prev_obs = observation
                observs.append(prepped_obs) # Observations will be used later

                # Take action
                prepped_obs = torch.from_numpy(prepped_obs).view(1,-1).float()
                t_state = Variable(prepped_obs)
                value, raw_output = net.forward(t_state, requires_value=True)
                action_pred = softmax(raw_output)
                pvec = sum_one(action_pred.data.tolist()[0]) # Create policy probability vector
                action = np.random.choice(action_dim, p=pvec) # Stochastically sample from vector
                old_pis.append(action_pred.data.squeeze()[action])
                actions.append(action) 

                observation, reward, done, info = env.step(action+2) # Add two for possible actions 2 or 3 (pong specific)
                
                value = value.data.squeeze()[0]
                if t != 0:
                    advantages.append(rewards[-1]+gamma*value-old_value) # Make advantage for previous step
                    mask.append(0) # Track rollout indices
                old_value = value
                rewards.append(reward)

                if done or t==n_tsteps-1 or rewards[-1] != 0: # Reached end of rollout for this episode

                    if rewards[-1] == 0: # Did not reach a terminal state (pong specific)
                        advantages.append(0) # use bootstrapped advantage (equivalent to V(t)-V(t))
                        rewards[-1] = old_value # Set bootstrapped reward for fitting critic later

                    else: # Reached terminal state (pong specific)
                        advantages.append(rewards[-1]-old_value) 
                        prev_obs = 0 # Reset for new rollout

                        # Reward book keeping
                        if reward_count < 100:
                            running_reward += rewards[-1]
                            reward_count += 1
                            avg_reward = running_reward/float(reward_count)
                        else:
                            avg_reward = .99*avg_reward + .01*rewards[-1]

                    mask.append(1) # Mark end of rollout
                    if done: 
                        observation = env.reset()

                    # Track observation for when we return to this environment in the next episode (danananana sup snoop!)
                    obs_bookmarks[i] = observation 
                    prev_bookmarks[i] = prev_obs
                    old_value = 0
                    break
                    
                    
    net.train(mode=True)
    print("T="+str(T),"– Episode", episode, "–– Avg Reward:", avg_reward, "–– Avg Action:", np.mean(actions))

    advantages = discount(advantages, gamma*lambda_, mask) # Generalized Value Estimation

    data = [actions, observs, rewards, old_pis, advantages, mask]
    net.fit_policy(data, optimizer, epochs=n_epochs, clip_const=clip_const, batch_size=fit_batch_size, entropy_const=entropy_const, val_const=val_const, gamma=gamma, lambda_=lambda_) 

    if episode % (ep_batch_size*5) == 0:
        torch.save(net.state_dict(), net_save_file)
        torch.save(optimizer.state_dict(), optim_save_file)

    # Check for memory leaks
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory Used: {:.2f} MB".format(max_mem_used / 1024))
    
    episode_reward = 0
    actions, observs, rewards, old_pis, advantages, mask = [], [], [], [], [], []
    net.train(mode=False)




