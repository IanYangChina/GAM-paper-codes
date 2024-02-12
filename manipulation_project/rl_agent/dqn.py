import time
import numpy as np
import torch as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.adam import Adam
from itertools import chain
import manipulation_project.rl_agent.utils.augmentation as aug
from manipulation_project.rl_agent.utils.networks_cnn import make_encoder, Critic
from drl_implementation.agent.agent_base import Agent
from drl_implementation.agent.utils.networks_mlp import Critic as SCritic
from drl_implementation.agent.utils.exploration_strategy import LinearDecayGreedy


class DQN(Agent):
    def __init__(self, algo_params, train_env, eval_env, create_logger=True, transition_tuple=None, path=None, seed=-1):
        # environment
        if train_env is not None:
            self.train_env = train_env
            self.train_env.seed(seed)
            self.num_train_env = self.train_env.num_env
        self.eval_env = eval_env
        self.eval_env.seed(seed)
        obs = self.eval_env.reset()
        if self.eval_env.image_obs:
            algo_params['state_shape'] = obs.shape
        else:
            algo_params['state_dim'] = obs.shape[0]
        algo_params.update({'action_dim': self.eval_env.action_space.n,
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        # training args
        self.training_timesteps = algo_params['training_timesteps']
        self.testing_episodes = algo_params['testing_episodes']
        self.testing_gap = algo_params['testing_gap']  # in timesteps
        self.saving_gap = algo_params['saving_gap']  # in timesteps
        self.log_interval = algo_params['log_interval']  # in timesteps

        algo_params['observation_normalization'] = False
        algo_params['actor_learning_rate'] = 0.0
        algo_params['discard_time_limit'] = False
        algo_params['tau'] = 1.0
        self.observation_normalization = algo_params['observation_normalization']
        super(DQN, self).__init__(algo_params,
                                  create_logger=create_logger,
                                  transition_tuple=transition_tuple,
                                  image_obs=self.eval_env.image_obs,
                                  action_type='discrete',
                                  path=path,
                                  training_mode='step_based',
                                  seed=seed)
        # torch
        if self.image_obs:
            encoder = make_encoder(image_size=self.eval_env.image_obs_size).to(self.device)
            self.network_dict.update({
                'encoder': encoder,
                'Q': Critic(out_dim=encoder.out_dim,
                            projection_dim=100,
                            state_shape=False,
                            action_dim=self.eval_env.action_space.n,
                            hidden_dim=1024).to(self.device),
                'Q_target': Critic(out_dim=encoder.out_dim,
                                   projection_dim=100,
                                   state_shape=False,
                                   action_dim=self.eval_env.action_space.n,
                                   hidden_dim=1024).to(self.device)
            })
            self.network_keys_to_save = ['encoder', 'Q_target']
            self.Q_1_optimizer = Adam(chain(self.network_dict['encoder'].parameters(),
                                            self.network_dict['Q'].parameters()),
                                      lr=self.critic_learning_rate)
            self._soft_update(self.network_dict['Q'], self.network_dict['Q_target'], tau=1)
            self.random_shift = aug.RandomShiftsAug(pad=4)
            self.random_color_jitter = aug.random_color_jitter
        else:
            self.network_dict.update({
                'Q_1': SCritic(self.state_dim, self.action_dim).to(self.device),
                'Q_1_target': SCritic(self.state_dim, self.action_dim).to(self.device),
                'Q_2': SCritic(self.state_dim, self.action_dim).to(self.device),
                'Q_2_target': SCritic(self.state_dim, self.action_dim).to(self.device),
            })
            self.network_keys_to_save = ['Q_1_target']
            self.Q_1_optimizer = Adam(self.network_dict['Q_1'].parameters(), lr=self.critic_learning_rate)
            self.Q_2_optimizer = Adam(self.network_dict['Q_2'].parameters(), lr=self.critic_learning_rate)
            self._soft_update(self.network_dict['Q_1'], self.network_dict['Q_1_target'], tau=1)
            self._soft_update(self.network_dict['Q_2'], self.network_dict['Q_2_target'], tau=1)

        self.exploration_strategy = LinearDecayGreedy(start=1.0, end=0.05,
                                                      decay=algo_params['exploration_decay_timesteps'], rng=self.rng)
        self.warmup_step = algo_params['warmup_step']
        self.clip_value = algo_params['clip_value']
        self.target_copy_gap = algo_params['target_copy_gap']
        # statistic dict
        self.tmp_q_loss, self.tmp_q_value, self.tmp_q_target_value = 0.0, 0.0, 0.0
        self.statistic_dict.update({
            'step_train_return': [],
            'episode_test_return': [],
            'q_value': [],
            'q_target_value': []
        })

    def run(self, render=False, test=False, load_network_step=None, sleep=0):
        if test:
            if render:
                import matplotlib as mlp
                mlp.use("TkAgg")
            ep_test_return = []
            num_episode = self.testing_episodes
            if load_network_step is not None:
                print("Loading network parameters...")
                self._load_network(step=load_network_step)
            print("Start testing...")
            for n in range(num_episode):
                ep_test_return.append(self._interact(render, test=True, sleep=sleep))
                # print("Episode %i" % n, "test return %0.2f" % ep_test_return[-1])
            print("Average return %0.2f" % (sum(ep_test_return) / self.testing_episodes))
            print("Average number of separation, object-dropped, exceeded-workspace & lift-failed: ", self.eval_env.num_dones[1:] / self.testing_episodes)
            print("Average scene alteration: ", self.eval_env.scene_alteration / self.testing_episodes)
            print("Average episode length: ", self.eval_env.num_ep_timesteps / self.testing_episodes)
            print("Finished testing...")
        else:
            print("Start training...")
            self._interact(render=render, test=False, sleep=0)
            print("Finished training, saving statistics...")
            self._save_statistics()

    def _interact(self, render=False, test=False, sleep=0):
        if test:
            if render and not self.eval_env.use_collected_states:
                f, axarr = plt.subplots(2, 2, figsize=(8, 8))
            done = False
            obs = self.eval_env.reset()
            if render:
                if self.eval_env.use_collected_states:
                    self.eval_env.render(mode='human')
                else:
                    rgb_0, depth_0 = self.eval_env.render(cam='sideview1')
                    rgb_1, depth_1 = self.eval_env.render(cam='sideview2')
                    axarr[0][0].imshow(rgb_0[::-1, :, :])
                    axarr[0][1].imshow(depth_0[::-1, :])
                    axarr[1][0].imshow(rgb_1[::-1, :, :])
                    axarr[1][1].imshow(depth_1[::-1, :])
                    plt.pause(0.00001)
            ep_return = 0
            # start a new episode
            while not done:
                action = self._select_action(obs, test=True)
                new_obs, reward, done, info = self.eval_env.step(action)
                if sleep > 0:
                    time.sleep(sleep)
                ep_return += reward
                obs = new_obs
                if render:
                    if self.eval_env.use_collected_states:
                        self.eval_env.render(mode='human')
                    else:
                        rgb_0, depth_0 = self.eval_env.render(cam='sideview1')
                        rgb_1, depth_1 = self.eval_env.render(cam='sideview2')
                        axarr[0][0].imshow(rgb_0[::-1, :, :])
                        axarr[0][1].imshow(depth_0[::-1, :])
                        axarr[1][0].imshow(rgb_1[::-1, :, :])
                        axarr[1][1].imshow(depth_1[::-1, :])
                        plt.pause(0.00001)
            # print(info)
            if render and not self.eval_env.use_collected_states:
                plt.close()
            return ep_return
        else:
            training_finished = False
            reward_tmp = []
            obs = self.train_env.reset()
            last_time = time.perf_counter()
            while not training_finished:
                if self.train_env.step_count >= self.warmup_step:
                    action = self._select_action(obs, test=False)
                else:
                    action = self.train_env.action_space.sample()
                new_obs, reward, done, info = self.train_env.step(action)
                reward_tmp.append(reward.sum() / self.num_train_env)  # average step reward
                self._remember_vec(obs, action, new_obs, reward, 1 - done, info)

                if self.train_env.step_count >= self.warmup_step:
                    self._learn(steps=self.num_train_env)

                if self.train_env.step_count % self.testing_gap == 0:
                    self.eval_env.num_dones *= 0
                    self.eval_env.num_ep_timesteps = 0
                    self.eval_env.distance = 0
                    self.eval_env.scene_alteration = 0
                    return_tmp = []
                    for n in range(self.testing_episodes):
                        ep_test_return = self._interact(render=render, test=True, sleep=sleep)
                        return_tmp.append(ep_test_return)
                    self.statistic_dict['episode_test_return'].append((sum(return_tmp) / self.testing_episodes))
                    print("Passed timesteps: %i, testing avg. return %0.3f" % (self.train_env.step_count,
                                                                               self.statistic_dict['episode_test_return'][-1]))
                    print("Passed clock time: %0.2f minutes" % ((time.perf_counter() - last_time) / 60))
                    last_time = time.perf_counter()
                    self.logger.add_scalar(tag='Test/return',
                                           scalar_value=self.statistic_dict['episode_test_return'][-1],
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/avg_num_ep_timesteps',
                                           scalar_value=self.eval_env.num_ep_timesteps / self.testing_episodes,
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/num_is_separated',
                                           scalar_value=self.eval_env.num_dones[1],
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/num_is_object_dropped',
                                           scalar_value=self.eval_env.num_dones[2],
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/num_is_exceeded_workspace',
                                           scalar_value=self.eval_env.num_dones[3],
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/avg_num_is_grasped',
                                           scalar_value=self.eval_env.num_grasped / self.testing_episodes,
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/avg_distance',
                                           scalar_value=self.eval_env.distance / self.testing_episodes,
                                           global_step=self.train_env.step_count)
                    self.logger.add_scalar(tag='Test/avg_scene_alteration',
                                           scalar_value=self.eval_env.scene_alteration / self.testing_episodes,
                                           global_step=self.train_env.step_count)

                if self.train_env.step_count % self.log_interval == 0:
                    self.statistic_dict['step_train_return'].append(sum(reward_tmp))  # return of 100 steps
                    reward_tmp.clear()
                    self._log()

                if self.train_env.step_count % self.saving_gap == 0:
                    self._save_network(step=self.train_env.step_count)

                if self.train_env.step_count > self.training_timesteps:
                    training_finished = True

                obs = new_obs

    def _log(self):
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_return',
                               scalar_value=self.statistic_dict['step_train_return'][-1],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_avg_ep_timesteps',
                               scalar_value=self.train_env.num_ep_timesteps / self.train_env.num_dones[0],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_num_episodes',
                               scalar_value=self.train_env.num_dones[0],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_avg_num_is_separated',
                               scalar_value=self.train_env.num_dones[1] / self.train_env.num_dones[0],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_avg_num_is_object_dropped',
                               scalar_value=self.train_env.num_dones[2] / self.train_env.num_dones[0],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_avg_num_is_exceeded_workspace',
                               scalar_value=self.train_env.num_dones[3] / self.train_env.num_dones[0],
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_num_grasped',
                               scalar_value=self.train_env.num_grasped,
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_distance',
                               scalar_value=self.train_env.distance,
                               global_step=self.train_env.step_count)
        self.logger.add_scalar(tag='Train/last_'+str(self.log_interval)+'step_scene_alteration',
                               scalar_value=self.train_env.scene_alteration,
                               global_step=self.train_env.step_count)
        self.train_env.num_dones *= 0
        self.train_env.num_ep_timesteps = 0
        self.train_env.distance = 0
        self.train_env.scene_alteration = 0

        if self.optim_step_count >= self.log_interval:
            self.logger.add_scalar(tag='Optimisation/q_loss',
                                   scalar_value=self.tmp_q_loss / self.log_interval,
                                   global_step=self.optim_step_count)
            self.logger.add_scalar(tag='Optimisation/q_value',
                                   scalar_value=self.tmp_q_value / self.log_interval,
                                   global_step=self.optim_step_count)
            self.logger.add_scalar(tag='Optimisation/target_q_value',
                                   scalar_value=self.tmp_q_target_value / self.log_interval,
                                   global_step=self.optim_step_count)
            self.tmp_q_loss, self.tmp_q_value, self.tmp_q_target_value = 0.0, 0.0, 0.0

    def _remember_vec(self, obs, action, new_obs, reward, done, info):
        for i in range(self.num_train_env):
            if info[i]['is_terminate']:
                new_obs[i] = info[i]['terminal_observation'].copy()
            self._remember(obs[i], action[i], new_obs[i], reward[i], done[i], new_episode=False)
            if self.observation_normalization:
                self.normalizer.store_history(new_obs[i])
        if self.observation_normalization:
            self.normalizer.update_mean()

    def _select_action(self, obs, test=False):
        if self.image_obs:
            if len(obs.shape) == 3:
                obs = np.array([obs])
            if test:
                obs = T.as_tensor(obs, device=self.device).permute(0, -1, 1, 2).float()
                with T.no_grad():
                    obs = self.network_dict['encoder'](obs[:, :3, :, :], obs[:, 3:6, :, :])
                    action = self.network_dict['Q_target'].get_action(obs, argmax_dim=1).tolist()[0]
                return action

            if self.exploration_strategy(self.env_step_count):
                return self.train_env.action_space.sample()
            else:
                obs = T.as_tensor(obs, device=self.device).permute(0, -1, 1, 2).float()
                with T.no_grad():
                    obs = self.network_dict['encoder'](obs[:, :3, :, :], obs[:, 3:6, :, :])
                    action = self.network_dict['Q_target'].get_action(obs, argmax_dim=1).tolist()
                return action
        else:
            obs = self.normalizer(obs)
            if len(obs.shape) == 1:
                obs = [obs]
            if test:
                obs = T.as_tensor(obs, dtype=T.float32, device=self.device)
                with T.no_grad():
                    # action = self.network_dict['Q_1_target'].get_action(obs)
                    values = self.network_dict['Q_1_target'](obs)
                    action = T.argmax(values, dim=1).tolist()
                return action

            if self.exploration_strategy(self.env_step_count):
                return self.train_env.action_space.sample()
            else:
                obs = T.as_tensor(obs, dtype=T.float32, device=self.device)
                with T.no_grad():
                    values = self.network_dict['Q_1_target'](obs)
                    action = T.argmax(values, dim=1).tolist()
                return action

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1), device=self.device)
                inds = None
            if self.image_obs:
                inputs = T.as_tensor(batch.state, device=self.device).permute(0, -1, 1, 2).float()
                actions = T.as_tensor(batch.action, dtype=T.long, device=self.device).unsqueeze(1)
                inputs_ = T.as_tensor(batch.next_state, device=self.device).permute(0, -1, 1, 2).float()
                rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
                done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

                with T.no_grad():
                    inputs_ = self.random_shift(inputs_)
                    inputs_ = self.random_color_jitter(inputs_)
                    inputs_ = self.network_dict['encoder'](inputs_[:, :3, :, :], inputs_[:, 3:6, :, :])
                    next_values_1, next_values_2 = self.network_dict['Q_target'](inputs_)
                    maximal_next_values_1 = next_values_1.max(1)[0].view(self.batch_size, 1)
                    maximal_next_values_2 = next_values_2.max(1)[0].view(self.batch_size, 1)
                    maximal_next_values = T.min(maximal_next_values_1, maximal_next_values_2)
                    value_target = rewards + done * self.gamma * maximal_next_values
                    value_target = T.clamp(value_target, self.clip_value[0], self.clip_value[1])

                self.Q_1_optimizer.zero_grad()
                inputs = self.random_shift(inputs)
                inputs = self.random_color_jitter(inputs)
                inputs = self.network_dict['encoder'](inputs[:, :3, :, :], inputs[:, 3:6, :, :])
                value_estimate_1, value_estimate_2 = self.network_dict['Q'](inputs)
                value_estimate_1 = value_estimate_1.gather(1, actions)
                value_estimate_2 = value_estimate_2.gather(1, actions)
                loss_1 = F.smooth_l1_loss(value_estimate_1, value_target.detach(), reduction='none')
                loss_2 = F.smooth_l1_loss(value_estimate_2, value_target.detach(), reduction='none')
                ((loss_1 * weights).mean() + (loss_2 * weights).mean()).backward()
                self.Q_1_optimizer.step()

                self.tmp_q_loss += loss_1.mean().detach()
                self.tmp_q_value += value_estimate_1.mean().detach()
                self.tmp_q_target_value += value_target.mean().detach()

                if self.prioritised:
                    assert inds is not None
                    self.buffer.update_priority(inds, np.abs(loss_1.cpu().detach().numpy()))

                self.optim_step_count += 1
                if self.optim_step_count % self.target_copy_gap == 0:
                    self._soft_update(self.network_dict['Q'], self.network_dict['Q_target'], tau=1)
            else:
                inputs = self.normalizer(batch.state)
                inputs = T.as_tensor(inputs, dtype=T.float32, device=self.device)
                actions = T.as_tensor(batch.action, dtype=T.long, device=self.device).unsqueeze(1)
                inputs_ = self.normalizer(batch.next_state)
                inputs_ = T.as_tensor(inputs_, dtype=T.float32, device=self.device)
                rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
                done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

                with T.no_grad():
                    maximal_next_values_1 = self.network_dict['Q_1_target'](inputs_).max(1)[0].view(self.batch_size, 1)
                    maximal_next_values_2 = self.network_dict['Q_2_target'](inputs_).max(1)[0].view(self.batch_size, 1)
                    maximal_next_values = T.min(maximal_next_values_1, maximal_next_values_2)
                    value_target = rewards + done * self.gamma * maximal_next_values
                    value_target = T.clamp(value_target, self.clip_value[0], self.clip_value[1])

                self.Q_1_optimizer.zero_grad()
                value_estimate_1 = self.network_dict['Q_1'](inputs).gather(1, actions)
                loss_1 = F.smooth_l1_loss(value_estimate_1, value_target.detach(), reduction='none')
                (loss_1 * weights).mean().backward()
                self.Q_1_optimizer.step()
                self.tmp_q_loss += loss_1.mean().detach()
                self.tmp_q_value += value_estimate_1.mean().detach()
                self.tmp_q_target_value += value_target.mean().detach()

                self.Q_2_optimizer.zero_grad()
                value_estimate_2 = self.network_dict['Q_2'](inputs).gather(1, actions)
                loss_2 = F.smooth_l1_loss(value_estimate_2, value_target.detach(), reduction='none')
                (loss_2 * weights).mean().backward()
                self.Q_2_optimizer.step()

                if self.prioritised:
                    assert inds is not None
                    self.buffer.update_priority(inds, np.abs(loss_1.cpu().detach().numpy()))

                self.optim_step_count += 1
                if self.optim_step_count % self.target_copy_gap == 0:
                    self._soft_update(self.network_dict['Q_1'], self.network_dict['Q_1_target'])
                    self._soft_update(self.network_dict['Q_2'], self.network_dict['Q_2_target'])
