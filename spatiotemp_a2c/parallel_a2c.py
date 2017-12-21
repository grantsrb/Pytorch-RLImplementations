import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gc
import resource
import sys
import spatiotemp_model as model
from utils import prep_obs, get_action, discount, reset, preprocess
from multiprocessing import Pool
import time


# hyperparameters
gamma = .99 # Discount factor
lambda_ = .97 # GAE moving average factor
batch_size = 3 # Number of times to perform rollout and collect gradients before updating model
n_envs = 20 # Number of environments to operate in parallel (note that this implementation does not run the environments on seperate threads)
n_processes = 3
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
spatio_const = 0.001 # Scales the spatiotemporal prediction loss function
max_norm = 0.4 # Scales the gradients using their norm
max_tsteps = 80e6 # The number of environmental steps to take before ending the algorithm
lr = 1e-4 # Divide by batchsize as a shortcut to averaging the gradient over multiple batches

print("gamma:", gamma)
print("lambda_:", lambda_)
print("batch_size:", batch_size)
print("n_envs:", n_envs)
print("n_tsteps:", n_tsteps)
print("val_const:", val_const)
print("entropy_const:", entropy_const)
print("spatio_const:", spatio_const)
print("max_norm:", max_norm)
print("lr:", lr)

batch_norm = False
predict_spatio = False
test = False
resume = False
render = False
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        str_arg = str(arg)
        if "spatio" in str_arg: predict_spatio = True
        if "norm" in str_arg: batch_norm = True
        if "test" in str_arg: test = True
        if "resume" in str_arg: resume = True
        if "render" in str_arg: render = True

print("Batch Norm:", batch_norm)
print("Spatio:", predict_spatio)
print("Test:", test)
print("Resume:", resume)
print("Render:", render)

if test:
    net_save_file = "net_test.p"
    optim_save_file = "optim_test.p"
    log_file = "log_test.txt"
elif predict_spatio:
   net_save_file = "net_spatio_pred.p"
   optim_save_file = "optim_spatio_pred.p"
   log_file = "log_spatio_pred.txt"
else:
   net_save_file = "net_control.p"
   optim_save_file = "optim_control.p"
   log_file = "log_control.txt"

# Make Pool
pool = Pool(n_processes)

# Make environments
envs = [gym.make("Pong-v0") for i in range(n_envs)]
for i,env in enumerate(envs):
    env.seed(i)

# Make model and optimizer
action_dim = 2 # Pong specific number of possible ep_actions
obs_bookmarks = [envs[i].reset() for i in range(n_envs)] # Used to track observations between environments
prepped_obs = preprocess(obs_bookmarks[0]) # Returns a prepped representation of the observation
prev_bookmarks = [np.zeros_like(prepped_obs) for i in range(n_envs)]
obs_shape = (prepped_obs.shape[0]*2,)+prepped_obs.shape[1:]
current_obses = [np.zeros(obs_shape,dtype=np.float32) for i in range(n_envs)]

# Various functions that will be useful later
logsoftmax = nn.LogSoftmax()
softmax = nn.Softmax()
mseloss = nn.MSELoss()

# Creat model and optimizer
net = model.Model(obs_shape, action_dim)
if torch.cuda.is_available():
    net = net.cuda()
    torch.FloatTensor = torch.cuda.FloatTensor
    torch.LongTensor = torch.cuda.LongTensor
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
    if batch_norm:
        logger.write("Batch Norm = True\n")
    else:
        logger.write("Batch Norm = False\n")
    logger.write("T, Avg Reward, Avg Loss, Norm, Action Loss, Value Loss, Entropy\n")

optimizer.zero_grad()

# Store ep_actions, observations, ep_values
ep_actions, observs, ep_rewards, ep_values, ep_advantages, ep_dones = [], [], [], [], [], []
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
    first_roll = True
    for t in range(n_tsteps):
        T+=n_envs

        if render: envs[0].render()

        # Prep observation
        async_arr = [pool.apply_async(prep_obs, [obs, prev]) for obs, prev in zip(obs_bookmarks, prev_bookmarks)]
        results = [async.get() for async in async_arr]
        prepped_obses, prev_bookmarks = zip(*results)
        prev_bookmarks = np.asarray(prev_bookmarks)
        observs.append(prepped_obses) # Observations will be used later

        # Take action
        if torch.cuda.is_available():
            t_state = Variable(torch.from_numpy(np.asarray(prepped_obses)).float().cuda())
        else:
            t_state = Variable(torch.from_numpy(np.asarray(prepped_obses)).float())
        values, raw_outputs, spatios, spatio_preds = net.forward(t_state, batch_norm=batch_norm)
        action_preds = softmax(raw_outputs).data.tolist()
        async_arr = [pool.apply_async(get_action, [pred, action_dim]) for pred in action_preds] # Create policy probability vector
        actions = [async.get() for async in async_arr]
        ep_actions.append(actions)

        # Add 2 to actions for possible actions 2 or 3 (pong specific)
        results = [envs[i].step(action+2) for i,action in enumerate(actions)]
        obs_bookmarks, rewards, dones, infos = zip(*results)
        obs_bookmarks = np.asarray(obs_bookmarks)
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)

        indices = (rewards!=0)
        if reward_count < 100:
            running_reward += np.sum(rewards[indices])
            reward_count += np.sum(indices)
            if reward_count > 0:
                avg_reward = running_reward/float(reward_count)
        else:
            if np.sum(indices) > 0:
                avg_reward = .99*avg_reward + .01*np.mean(rewards[indices])

        values = values.data.squeeze().cpu().numpy()
        if not first_roll:
            ep_advantages.append(ep_rewards[-1]+gamma*values-ep_values[-1]) # Make advantage for previous step
        else:
            first_roll = False
        ep_values.append(values)
        ep_rewards.append(rewards)
        ep_dones.append(dones)

        for i,d in enumerate(dones):
            if d:
                obs_bookmarks[i] = envs[i].reset()
                prev_bookmarks[i] = np.zeros_like(prev_bookmarks[i])




    # End of rollout
    bootstrap_indices = (ep_rewards[-1] == 0) # Pong specific
    ep_rewards[-1][bootstrap_indices] = ep_values[-1][bootstrap_indices] # Bootstrap
    ep_advantages.append(ep_rewards[-1]-ep_values[-1])


    net.train(mode=True)
    episode += 1
    print("T="+str(T),"– Episode", episode, "–– Avg Reward:", avg_reward, "–– Avg Action:", np.mean(ep_actions))

    if reward_count > 100 and avg_reward > rew_cutoff:
        rew_cutoff = rew_cutoff + 0.1
        entropy_const = entropy_const*.75
        max_norm *= .99
        lr *= .9
        optimizer.lr = lr

    ep_dones = np.asarray(ep_dones).astype(np.int)
    ep_rewards = np.asarray(ep_rewards).astype(np.float32)
    async_arr = [pool.apply_async(discount, [ep_rewards[:,i], gamma, ep_dones[:,i]]) for i in range(n_envs)]
    rewards = [async.get() for async in async_arr]
    t_rewards = Variable(torch.FloatTensor(rewards).view(-1))

    observs = np.asarray(observs).astype(np.float32).swapaxes(0,1).reshape((-1,)+obs_shape)
    if torch.cuda.is_available():
        t_observs = Variable(torch.from_numpy(observs).cuda())
    else:
        t_observs = Variable(torch.from_numpy(observs))

    values, raw_actions, spatios, spatio_preds = net.forward(t_observs, batch_norm=batch_norm)
    spatio_preds = spatio_preds.view(spatio_preds.size(0),-1)
    spatios = Variable(spatios.data.view(spatios.size(0), -1))
    softlogs = logsoftmax(raw_actions)
    actions = torch.LongTensor(ep_actions).permute(1,0).contiguous().view(-1)
    if torch.cuda.is_available():
        action_logs = softlogs[torch.arange(0,len(actions)).long().cuda(),actions] # Only want action predictions for the performed ep_actions
    else:
        action_logs = softlogs[torch.arange(0,len(actions)).long(),actions]

    advantages = np.asarray(ep_advantages).astype(np.float32)
    async_arr = [pool.apply_async(discount, [advantages[:,i], gamma*lambda_,ep_dones[:,i]]) for i in range(n_envs)]
    advantages = [async.get() for async in async_arr]
    advantages = Variable(torch.FloatTensor(advantages).view(-1))
    #returns = Variable(ep_advantages.data + ep_values.data.squeeze())

    mask = torch.from_numpy(1.-ep_dones).float()
    if torch.cuda.is_available():
        mask = mask.cuda()
    mask = Variable(mask.permute(1,0).contiguous().view(-1))


    # Evaluate Losses
    action_loss = -torch.mean(action_logs*advantages) # Standard policy gradient
    value_loss = val_const*mseloss(values.squeeze(), t_rewards)
    entropy = -entropy_const*torch.mean(softmax(raw_actions)*softlogs)
    if predict_spatio:
        spatio_loss = spatio_const*torch.mean(torch.sum((spatio_preds[:-1]-spatios[1:])**2, dim=-1)*mask[:-1])
        loss = action_loss + value_loss - entropy + spatio_loss
    else:
        loss = action_loss + value_loss - entropy

    loss.backward()
    net.check_grads() # Checks for NaN in gradients. Pytorch can be finicky



    avg_loss = loss.data.squeeze()[0] if avg_loss == None else .99*avg_loss + .01*loss.data.squeeze()[0]

    if episode % batch_size == 0:
        epoch += 1
        norm = nn.utils.clip_grad_norm(net.parameters(), max_norm) # Reduces gradients if their norm is too big. This prevents large changes to the policy. Large changes can cripple learning for the policy.
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch", epoch, "–– Avg Reward:", avg_reward, "–– Avg Loss:",avg_loss, "–– Norm:", norm)
        if predict_spatio:
            print("Act Loss:", action_loss.data[0], "–– Val Loss:", value_loss.data[0], "\nEntropy",entropy.data[0], "–– Spatio Loss:", spatio_loss.data[0])
        else:
            print("Act Loss:", action_loss.data[0], "–– Val Loss:", value_loss.data[0], "\nEntropy",entropy.data[0])
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
    ep_actions, observs, ep_rewards, ep_values, ep_advantages, ep_dones = [], [], [], [], [], []
    net.train(mode=False)

logger.close()
