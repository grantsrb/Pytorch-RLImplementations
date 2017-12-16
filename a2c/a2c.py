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
gamma = .99 # Discount factor
lambda_ = .97 # GAE moving average factor
batch_size = 3 # Number of times to perform rollout and collect gradients before updating model
n_envs = 20 # Number of environments to operate in parallel (note that this implementation does not run the environments on seperate threads)
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
max_norm = 0.5 # Scales the gradients using their norm
max_tsteps = 80e6 # The number of environmental steps to take before ending the algorithm
lr = 1e-2/batch_size/n_envs # Divide by batchsize as a shortcut to averaging the gradient over multiple batches

net_save_file = "net_state_dict.p"
optim_save_file = "optim_state_dict.p"

resume = False
render = False
if len(sys.argv) > 1:
    resume = bool(sys.argv[1])
    if len(sys.argv) > 2:
        render = bool(sys.argv[2])


def preprocess(pic):
    """
    Preprocesses the observations for improved learning. (Stolen from Karpathy's blog)
    """
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic.ravel()

def discount(rs, disc_factor, mask):
    """
    Discounts the rewards or advantages. mask is an array used to distinguish rollouts.
    """
    discounteds = [0]*len(rs)
    running_sum = 0
    for i in reversed(range(len(rs))):
        if mask[i] == 1: running_sum = 0
        running_sum = running_sum*disc_factor + rs[i]
        discounteds[i] = running_sum
    return discounteds

def sum_one(action_vec):
    """
    Ensures values in action_vec sum to 1. Pytorch softmax was returning values that did not quite sum to 1.
    """
    new_vec = [0]*len(action_vec)
    running_sum = 0
    for i in range(len(action_vec)-1):
        new_vec[i] = round(action_vec[i], 4)
        running_sum += new_vec[i]
    new_vec[-1] = 1-running_sum
    return new_vec
    
# Make environments
envs = [gym.make("Pong-v0") for i in range(n_envs)]
for i,env in enumerate(envs):
    env.seed(i)
obs_bookmarks = [env.reset() for env in envs] # Used to track observations between environments
prev_bookmarks = [0 for i in range(n_envs)]

# Make model and optimizer
action_dim = 2 # Pong specific number of possible actions
prepped_state = preprocess(obs_bookmarks[0]) # Returns a vector representation of the observation
net = model.Model(prepped_state.shape[0], action_dim) 
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
actions, observs, rewards, values, advantages, mask = [], [], [], [], [], []
episode_reward = 0
avg_reward = 0
running_reward = 0
reward_count = 0
avg_loss = None
episode, epoch = 0, 0
T = 0 # Tracks total environmental steps taken

net.train(mode=False)

while T < max_tsteps:

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
            actions.append(action) 

            observation, reward, done, info = env.step(action+2) # Add two for possible actions 2 or 3 (pong specific)
            
            value = value.data.squeeze()[0]
            if t != 0:
                advantages.append(rewards[-1]+gamma*value-values[-1]) # Make advantage for previous step
                mask.append(0) # Track rollout indices
            values.append(value)
            rewards.append(reward)

            if done or t==n_tsteps-1 or rewards[-1] != 0: # Reached end of rollout for this episode

                if rewards[-1] == 0: # Did not reach a terminal state (pong specific)
                    advantages.append(0) # use bootstrapped reward equivalent to V(t)-V(t)
                    rewards[-1] = values[-1] # Set bootstrapped reward to fit critic later

                else: # Reached terminal state
                    advantages.append(rewards[-1]-values[-1]) 

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
                    prev_obs = 0

                # Track observation for when we return to this environment in the next episode (danananana sup snoop!)
                obs_bookmarks[i] = observation 
                prev_bookmarks[i] = prev_obs
                break
                
                
    net.train(mode=True)
    episode += 1
    print("T="+str(T),"– Episode", episode, "–– Avg Reward:", avg_reward, "–– Avg Action:", np.mean(actions))

    rewards = discount(rewards, gamma, mask) # Discount rewards
    t_rewards = Variable(torch.FloatTensor(rewards))

    advantages = discount(advantages, gamma*lambda_, mask) # Generalized Value Estimation
    advantages = Variable(torch.FloatTensor(advantages))

    t_observs = Variable(torch.from_numpy(np.asarray(observs)).float())
    values, raw_actions = net.forward(t_observs, requires_value=True)
    softlogs = logsoftmax(raw_actions)
    action_logs = softlogs[list(range(len(actions))),actions] # Only want action predictions for the performed actions
    action_loss = -torch.mean(action_logs*advantages) # Standard policy gradient

    value_loss = val_const*mseloss(values.squeeze(), t_rewards)
 
    entropy = -entropy_const*torch.mean(softmax(raw_actions)*softlogs) 

    """
    Entropy is used to discourage the policy from giving too strong of probability 
    to a single action. Without the entropy term, the policy can quickly collapse 
    to favoring a single action and never learn to do any better. The entropy is 
    a measure of randomness and takes it's largest values when the action probabilities 
    are equally distributed. Thus we want to maximize the entropy in our loss function. 
    We're using gradient descent which means that we want to subtract our entropy term 
    from the loss. I had this wrong for a while... Hopefully this comment saves another
    from the same fate!
    """

    loss = action_loss + value_loss - entropy
    loss.backward()
    net.check_grads() # Checks for NaN in gradients.
    
    avg_loss = loss.data.squeeze()[0] if avg_loss == None else .99*avg_loss + .01*loss.data.squeeze()[0]

    if episode % batch_size == 0:
        epoch += 1
        norm = nn.utils.clip_grad_norm(net.parameters(), max_norm) # Reduces gradients if their norm is too big. This prevents large changes to the policy. Large changes can cripple learning for the policy.
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch", epoch, "–– Avg Reward:", avg_reward, "–– Avg Loss:",avg_loss, "–– Norm:", norm)
        print("Act Loss:", action_loss.data[0], "– Val Loss:", value_loss.data[0], "– Entropy",entropy.data[0])
        torch.save(net.state_dict(), net_save_file)
        torch.save(optimizer.state_dict(), optim_save_file)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory Used: {:.2f} MB".format(max_mem_used / 1024))
    
    episode_reward = 0
    actions, observs, rewards, values, advantages, mask = [], [], [], [], [], []
    net.train(mode=False)











