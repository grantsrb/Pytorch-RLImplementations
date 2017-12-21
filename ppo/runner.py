import torch
import model
import utils
from multiprocessing import Pool

class Runner():
    """
    Runner is used to collect a rollout of data from the environment. Inspired by OpenAI's baselines.
    """

    def __init__(self, env_name, n_envs, n_processes=3):
        self.envs = [gym.make(env_name).seed(i) for i in range(n_envs)]
        self.obs_bookmarks = [self.envs[i].reset() for i in range(n_envs)]
        obs_shape = utils.preprocess(self.obs_bookmarks[0]).shape
        self.prev_bookmarks = [np.zeros(obs_shape).astype(np.float32) for i in range(n_envs)]
        self.obs_shape = [obs_shape[0]*2]+obs_shape[1:]
        self.avg_reward = 0
        self.running_reward = 0
        self.reward_count = 0
        self.avg_loss = None
        self.episode = 0
        self.epoch = 0
        self.T = 0 # Tracks total environmental steps taken


    def rollout(self, net, t_steps, render=False):
        ep_actions, ep_observs, ep_rewards, ep_values, ep_deltas, ep_dones, ep_pis = [], [], [], [], [], [], []
        first_roll = True
        for i in range(t_steps):
            self.T+=n_envs # Tracks total number of frames
            if render: self.envs[0].render()

            # Prep observation
            async_arr = [pool.apply_async(prep_obs, [obs, prev]) for obs, prev in zip(self.obs_bookmarks, self.prev_bookmarks)]
            results = [async.get() for async in async_arr]
            prepped_obses, prev_bookmarks = zip(*results)
            self.prev_bookmarks = np.asarray(prev_bookmarks)
            ep_observs.append(prepped_obses)

            # Take action
            if torch.cuda.is_available():
                t_obses = Variable(torch.from_numpy(np.asarray(prepped_obses)).float().cuda())
            else:
                t_obses = Variable(torch.from_numpy(np.asarray(prepped_obses)).float())
            values, raw_outputs, spatios, spatio_preds = net.forward(t_obses, batch_norm=batch_norm)
            action_preds = softmax(raw_outputs).data.tolist()
            ep_pis.append(action_preds)
            async_arr = [pool.apply_async(get_action, [pred, action_dim]) for pred in action_preds] # Sample action from policy's distribution
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
            if reward_count < 100:
                running_reward += np.sum(rewards[r_indices])
                reward_count += np.sum(r_indices)
                if reward_count > 0:
                    avg_reward = running_reward/float(reward_count)
            else:
                if np.sum(r_indices) > 0:
                    avg_reward = .99*avg_reward + .01*np.mean(rewards[r_indices])

            # Collect deltas for GAE later
            values = values.data.squeeze().cpu().numpy()
            if not first_roll:
                ep_deltas.append(ep_rewards[-1]+gamma*values-ep_values[-1])
            else:
                first_roll = False
            ep_values.append(values)
            ep_rewards.append(rewards)
            ep_dones.append(dones)

            for i,d in enumerate(dones):
                if d:
                    self.obs_bookmarks[i] = self.envs[i].reset()
                    self.prev_bookmarks[i] = np.zeros_like(self.prev_bookmarks[i])

        return ep_actions, ep_observs, ep_rewards, ep_values, ep_deltas, ep_dones, ep_pis









#
