"""
A collection of useful functions and classes
"""

import numpy as np

def preprocess(pic):
    """
    Preprocesses the observations from Pong for improved learning. (Stolen from Karpathy's blog)

    pic - numpy array
    """
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic[None]

def prep_obs(observation, prev_observation):
    """
    Creates a prepped observation combined with the previous observation
    """
    observation = preprocess(observation)
    obs_shape = (observation.shape[0]*2,)+observation.shape[1:]
    prepped_obs = np.zeros(obs_shape)
    prepped_obs[:obs_shape[0]//2] = observation
    prepped_obs[obs_shape[0]//2:] = prev_observation
    prev_observation = observation
    return prepped_obs, prev_observation

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

def get_action(action_vec, action_dim, samp_type="pg", rand_sample=0.1):
    """
    Samples action from action_vec.

    samp_type - distinguishes between policy gradients and q values
    rand_sample - probability to randomly sample q value
    """
    if "pg" in samp_type or "pol" in samp_type:
        p_vec = sum_one(action_vec) # Used to solve sum errors in numpy.random.choice
        action = np.random.choice(action_dim, p=p_vec) # Stochastically sample from vector
    else:
        prob = np.random.random()
        if prob <= rand_sample:
            action = np.random.choice(action_dim)
        else:
            action = np.argmax(action_vec)
    return int(action)

def gae(rewards, values, mask, gamma, lambda_):
    """
    Calculates gae advantages using the provided rewards and values
    """
    advantages = [0]*len(rewards)
    running_sum = 0
    for i in range(len(rewards)):
        if mask[i] == 1:
            if rewards[i] != 1:
                running_sum = 0 # Bootstrap
                rewards[i] = values[i]
            else:
                running_sum = rewards[i]-values[i]
        else:
            running_sum = gamma*lambda_*running_sum + (rewards[i]+gamma*values[i+1]-values[i])
        advantages[i] = running_sum
    return advantages, rewards
