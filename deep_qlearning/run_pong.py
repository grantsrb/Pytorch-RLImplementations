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
from utils import prep_obs, get_action, discount, preprocess
from multiprocessing import Pool
from runner import Runner
from fitter import Fitter


# hyperparameters
gamma = .99 # Discount factor
lambda_ = .97 # GAE moving average factor
batch_size = 3 # Number of times to perform rollout and collect gradients before updating model
n_envs = 20 # Number of environments to operate in parallel (note that this implementation does not run the environments on seperate threads)
n_processes = n_envs
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
spatio_const = 0.001 # Scales the spatiotemporal prediction loss function
max_norm = 0.4 # Scales the gradients using their norm
max_tsteps = 80e6 # The number of environmental steps to take before ending the algorithm
lr = 1e-4 # Divide by batchsize as a shortcut to averaging the gradient over multiple batches
# PPO Specific
n_epochs = 5
clip_const = 0.2

if n_tsteps*n_envs % batch_size != 0:
    while n_tsteps*n_envs % batch_size != 0:
        batch_size+=1
    print("Batch size is not perfectly divisible. Automatically switched to:",batch_size)
ppo_batch_size = n_tsteps*n_envs//batch_size

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
print("n_epochs", n_epochs)
print("clip_const", clip_const)
print("ppo_batch_size", ppo_batch_size)

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

# Make data collection object
env_name = "Pong-v0"
runner = Runner(env_name, n_envs, pool)

# Create model and optimizer
action_dim = 2 # Pong specific number of possible actions
net = model.Model(runner.obs_shape, action_dim, batch_norm=batch_norm)
if torch.cuda.is_available():
    net = net.cuda()
    torch.FloatTensor = torch.cuda.FloatTensor
    torch.LongTensor = torch.cuda.LongTensor
optimizer = optim.Adam(net.parameters(), lr=lr)

q_fitter = Fitter(net, optimizer, pool, val_const=val_const, entropy_const=entropy_const,
                        spatio_const=spatio_const, gamma=gamma, lambda_=lambda_,
                        predict_spatio=predict_spatio)

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
    if predict_spatio:
        logger.write("T, Avg Reward, Avg Loss, Action Loss, Value Loss, Entropy, Spatio\n")
    else:
        logger.write("T, Avg Reward, Avg Loss, Action Loss, Value Loss, Entropy\n")

optimizer.zero_grad()
rew_cutoff = -0.9

net.train(mode=False)
avg_loss = None
epoch = 0

while runner.T < max_tsteps:

    collections = []
    for i in range(batch_size):
        data = runner.rollout(net, n_tsteps, batch_norm, gamma, render)
        collections.append(data)

    combined_data = []
    for i in range(len(data)):
        if i == 0:
            temp = []
            for j in range(batch_size):
                temp += collections[j][i]
        else:
            temp = np.concatenate([collections[j][i] for j in range(batch_size)], axis=0)
        combined_data.append(temp)

    net.train(mode=True)
    print("T="+str(runner.T),"– Episode", runner.episode, "–– Avg Reward:", runner.avg_reward, "–– Avg Action:", np.mean(combined_data[0]))

    if runner.reward_count > 100 and runner.avg_reward > rew_cutoff:
        rew_cutoff = rew_cutoff + 0.1
        entropy_const = entropy_const*.75
        max_norm *= .99
        lr *= .9
        optimizer.lr = lr

    # Book keeping
    epoch += 1
    losses = q_fitter.fit(combined_data, epochs=n_epochs, batch_size=ppo_batch_size, clip_const=clip_const, max_norm=max_norm)
    loss, action_loss, value_loss, entropy, spatio_loss = losses

    avg_loss = loss if avg_loss == None else .99*avg_loss + .01*loss

    print("Epoch", epoch, "–– Avg Loss:", avg_loss)
    if predict_spatio:
        print("Act Loss:", action_loss, "–– Val Loss:", value_loss, "\nEntropy",entropy, "–– Spatio Loss:", spatio_loss)
        seq = [runner.T, runner.avg_reward, avg_loss, action_loss, value_loss, entropy, spatio_loss]
    else:
        print("Act Loss:", action_loss, "–– Val Loss:", value_loss, "\nEntropy",entropy)
        seq = [runner.T, runner.avg_reward, avg_loss, action_loss, value_loss, entropy]

    logger.write(",".join([str(x) for x in seq]))
    logger.write("\n")
    logger.flush()
    torch.save(net.state_dict(), net_save_file)
    torch.save(optimizer.state_dict(), optim_save_file)

    # Check for memory leaks
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory Used: {:.2f} MB".format(max_mem_used / 1024))

    net.train(mode=False)

logger.close()
