import torch
from torch.autograd import Variable
import numpy as np
import spatiotemp_model as model
import utils
from multiprocessing import Pool
import gym

class Runner():
    """
    Runner is used to collect a rollout of data from the environment. Inspired by OpenAI's baselines.
    """

    def __init__(self, env_name, n_envs, pool, rand_sample=0.1):
        self.n_envs = n_envs
        self.envs = [gym.make(env_name) for i in range(n_envs)]
        for i in range(n_envs):
            self.envs[i].seed(i)
        self.obs_bookmarks = [self.envs[i].reset() for i in range(n_envs)]
        obs_shape = utils.preprocess(self.obs_bookmarks[0]).shape
        self.prev_bookmarks = [np.zeros(obs_shape).astype(np.float32) for i in range(n_envs)]
        self.obs_shape = (obs_shape[0]*2,)+obs_shape[1:]
        self.pool = pool
        self.avg_reward = 0
        self.running_reward = 0
        self.reward_count = 0
        self.episode = 0
        self.T = 0 # Tracks total environmental steps taken
        self.rand_sample_prob = rand_sample



    def rollout(self, net, n_tsteps, batch_norm, gamma=0.99, render=False):
        ep_actions, ep_observs, ep_rewards, ep_values, ep_deltas, ep_dones = [], [], [], [], [], []
        first_roll = True
        self.episode += 1
        for i in range(n_tsteps):
            self.T+=self.n_envs # Tracks total number of frames
            if render: self.envs[0].render()

            # Prep observation
            async_arr = [self.pool.apply_async(utils.prep_obs, [obs, prev]) for obs, prev in zip(self.obs_bookmarks, self.prev_bookmarks)]
            results = [async.get() for async in async_arr]
            prepped_obses, prev_bookmarks = zip(*results)
            self.prev_bookmarks = np.asarray(prev_bookmarks)
            ep_observs.append(prepped_obses)

            # Take action
            if torch.cuda.is_available():
                t_obses = Variable(torch.from_numpy(np.asarray(prepped_obses)).float().cuda())
            else:
                t_obses = Variable(torch.from_numpy(np.asarray(prepped_obses)).float())
            values, raw_outputs, spatios, spatio_preds = net.forward(t_obses)
            action_preds = net.softmax(raw_outputs).data.tolist()
            async_arr = [self.pool.apply_async(utils.get_action, [pred, net.act_space, "q-learning", self.rand_sample_prob]) for pred in action_preds]
            actions = [async.get() for async in async_arr]
            ep_actions.append(actions)

            # Add 2 to actions for possible actions 2 or 3 (pong specific)
            results = [self.envs[i].step(action+2) for i,action in enumerate(actions)]
            obs_bookmarks, rewards, dones, infos = zip(*results)
            self.obs_bookmarks = np.asarray(obs_bookmarks)
            dones = np.asarray(dones)
            rewards = np.asarray(rewards)

            # Track average reward
            r_indices = (rewards!=0)
            if self.reward_count < 100:
                self.running_reward += np.sum(rewards[r_indices])
                self.reward_count += np.sum(r_indices)
                if self.reward_count > 0:
                    self.avg_reward = self.running_reward/float(self.reward_count)
            else:
                if np.sum(r_indices) > 0:
                    self.avg_reward = .99*self.avg_reward + .01*np.mean(rewards[r_indices])

            # Collect deltas for GAE later where delta = r(t) + gamma*V(t+1) - V(t)
            values = values.data.squeeze().cpu().numpy()
            if not first_roll:
                ep_deltas.append(ep_rewards[-1]+gamma*values-ep_values[-1])
            else:
                first_roll = False
            ep_values.append(values)
            ep_rewards.append(rewards)
            ep_dones.append(dones)

            # Reset environments that are finished
            for i,d in enumerate(dones):
                if d:
                    self.obs_bookmarks[i] = self.envs[i].reset()
                    self.prev_bookmarks[i] = np.zeros_like(self.prev_bookmarks[i])

        # End of rollout bootstrapping
        ep_dones[-1][:] = 1. # Track the end of the rollout
        bootstrap_indices = (ep_rewards[-1] == 0) # Pong specific
        ep_rewards[-1][bootstrap_indices] = ep_values[-1][bootstrap_indices] # Bootstrap
        ep_deltas.append(ep_rewards[-1]-ep_values[-1])


        ep_observs = np.asarray(ep_observs, dtype=np.float32)
        ep_rewards = np.asarray(ep_rewards, dtype=np.float32)
        ep_values = np.asarray(ep_values, dtype=np.float32)
        ep_deltas = np.asarray(ep_deltas, dtype=np.float32)
        ep_dones = np.asarray(ep_dones, dtype=np.int)

        return ep_actions, ep_observs, ep_rewards, ep_values, ep_deltas, ep_dones









#
