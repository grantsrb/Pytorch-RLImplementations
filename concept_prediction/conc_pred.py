import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gc
import resource
import sys
import cnn_model as model
import utils


# hyperparameters
gamma = .99 # Discount factor
lambda_ = .97 # GAE moving average factor
batch_size = 3 # Number of times to perform rollout and collect gradients before updating model
n_envs = 20 # Number of environments to operate in parallel (note that this implementation does not run the environments on seperate threads)
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
conc_const = 0.1
max_norm = 0.4 # Scales the gradients using their norm
max_tsteps = 80e6 # The number of environmental steps to take before ending the algorithm
lr = 1e-3/n_envs # Divide by batchsize as a shortcut to averaging the gradient over multiple batches
batch_norm = True
predict_concept = True


resume = False
render = False
if len(sys.argv) > 1:
    idx = 1
    if "conc" in str(sys.argv[idx]):
        predict_concept = True
        idx += 1
    if len(sys.argv) > idx:
        resume = bool(sys.argv[idx])
        idx += 1
        if len(sys.argv) > idx:
            render = bool(sys.argv[idx])

if predict_concept:
    net_save_file = "net_conc_pred.p"
    optim_save_file = "optim_conc_pred.p"
    log_file = "log_conc_pred.txt"
else:
    net_save_file = "net_control.p"
    optim_save_file = "optim_control.p"
    log_file = "log_control.txt"

# Make environments
envs = [gym.make("Pong-v0") for i in range(n_envs)]
for i,env in enumerate(envs):
    env.seed(i)

# Make model and optimizer
action_dim = 2 # Pong specific number of possible actions
obs_bookmarks = [env.reset() for env in envs] # Used to track observations between environments
prepped_obs = utils.preprocess(obs_bookmarks[0]) # Returns a vector representation of the observation
prev_bookmarks = [np.zeros_like(prepped_obs) for i in range(n_envs)]
obs = np.zeros((prepped_obs.shape[0]*2,)+prepped_obs.shape[1:],dtype=np.float32)

# Various functions that will be useful later
logsoftmax = nn.LogSoftmax()
softmax = nn.Softmax()
mseloss = nn.MSELoss()

# Creat model and optimizer
net = model.Model(obs.shape, action_dim) 
if torch.cuda.is_available():
    net = net.cuda()
    torch.FloatTensor = torch.cuda.FloatTensor
    logsoftmax = logsoftmax.cuda()
    softmax = softmax.cuda()
    mseloss = mseloss.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)

if resume:
    net.load_state_dict(torch.load(net_save_file))
    optimizer.load_state_dict(torch.load(optim_save_file))
    logger = open(log_file, 'a+')
else:
    logger = open(log_file, 'w+')
    logger.write("T, Avg Reward, Avg Loss, Norm, Action Loss, Value Loss, Entropy\n")

optimizer.zero_grad()


# Store actions, observations, values
actions, observs, rewards, values, advantages, mask, concepts, conc_preds = [], [], [], [], [], [], [], []
episode_reward = 0
avg_reward = 0
running_reward = 0
reward_count = 0
avg_loss = None
episode, epoch = 0, 0
rew_cutoff = -0.8
T = 0 # Tracks total environmental steps taken

net.train(mode=False)

while T < max_tsteps:

    for i,env in enumerate(envs):
        observation = obs_bookmarks[i]
        prev_obs = prev_bookmarks[i]
        first_roll = True
        
        for t in range(n_tsteps):
            T+=1

            if render and i == 0: env.render()

            # Prep observation
            observation = utils.preprocess(observation)
            prepped_obs = np.concatenate([observation, prev_obs], axis=0)
            prev_obs = observation
            observs.append(prepped_obs.tolist()) # Observations will be used later

            # Take action
            prepped_obs = torch.FloatTensor(prepped_obs.tolist()).unsqueeze(0)
            t_state = Variable(prepped_obs)
            value, raw_output, concept, conc_pred = net.forward(t_state, batch_norm=batch_norm)
            action_pred = softmax(raw_output)
            pvec = utils.sum_one(action_pred.data.tolist()[0]) # Create policy probability vector
            action = np.random.choice(action_dim, p=pvec) # Stochastically sample from vector
            actions.append(action) 

            observation, reward, done, info = env.step(action+2) # Add two for possible actions 2 or 3 (pong specific)
            
            value = value.data.squeeze()[0]
            if not first_roll: 
                advantages.append(rewards[-1]+gamma*value-values[-1]) # Make advantage for previous step
                mask.append(0) # Track rollout indices
            else:
                first_roll = False
            values.append(value)
            rewards.append(reward)
            concepts.append(concept.data.view(1,-1))

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
                    prev_obs = np.zeros_like(prev_obs)

                # Track observation for when we return to this environment in the next episode (danananana sup snoop!)
                obs_bookmarks[i] = observation 
                prev_bookmarks[i] = prev_obs
                first_roll = True
                
                
    net.train(mode=True)
    episode += 1
    print("T="+str(T),"– Episode", episode, "–– Avg Reward:", avg_reward, "–– Avg Action:", np.mean(actions))

    if reward_count > 100 and avg_reward > rew_cutoff:
        rew_cutoff = rew_cutoff + 0.1
        entropy_const = entropy_const*.66
        max_norm *= .95
        lr *= .9
        optimizer.lr = lr

    rewards = utils.discount(rewards, gamma, mask) # Discount rewards
    t_rewards = Variable(torch.FloatTensor(rewards))

    t_observs = Variable(torch.FloatTensor(observs))
    values, raw_actions, _, conc_preds = net.forward(t_observs, batch_norm=batch_norm)
    conc_preds = conc_preds.view(conc_preds.size(0),-1)
    softlogs = logsoftmax(raw_actions)
    action_logs = softlogs[list(range(len(actions))),actions] # Only want action predictions for the performed actions

    advantages = utils.discount(advantages, gamma*lambda_, mask) # Generalized Value Estimation
    advantages = Variable(torch.FloatTensor(advantages))
    #returns = Variable(advantages.data + values.data.squeeze())
    concepts = Variable(torch.cat(concepts,0))
    mask = torch.FloatTensor(mask)
    ones = mask.new(mask.size()) + 1.0
    mask = Variable(ones-mask)

    action_loss = -torch.mean(action_logs*advantages) # Standard policy gradient

    value_loss = val_const*mseloss(values.squeeze(), t_rewards)
 
    entropy = -entropy_const*torch.mean(softmax(raw_actions)*softlogs) 

    if predict_concept: 
        conc_loss = conc_const*torch.mean(torch.sum((conc_preds[:-1]-concepts[1:])**2, dim=-1)*mask[:-1])
        loss = action_loss + value_loss - entropy + conc_loss
    else:
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
        seq = [T, avg_reward, avg_loss, norm, action_loss.data[0], value_loss.data[0], entropy.data[0]]
        logger.write(",".join([str(x) for x in seq]))
        logger.write("\n")
        logger.flush()
        torch.save(net.state_dict(), net_save_file)
        torch.save(optimizer.state_dict(), optim_save_file)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory Used: {:.2f} MB".format(max_mem_used / 1024))
    
    episode_reward = 0
    actions, observs, rewards, values, advantages, mask, concepts, conc_preds = [], [], [], [], [], [], [],[]
    net.train(mode=False)

logger.close()

