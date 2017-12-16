"""
A collection of useful functions and classes
"""

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
    
