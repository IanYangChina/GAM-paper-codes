import numpy as np
from gym import spaces
import multiprocessing as mp
import importlib
from manipulation_project.env.sg_data_loader import DataLoader

graspnet_module_name = 'manipulation_project.graspnet_agent.grasp_estimator'


class SingleEnvWrapper(object):
    def __init__(self, env_class, config, grasp_filter=None):
        self.image_obs = config['image_obs']
        self.image_obs_size = config['image_obs_size']
        self.num_discarded_grasps = 0
        self.num_dones = np.zeros(shape=(5,))
        self.num_ep_timesteps = 0
        self.distance = 0
        self.scene_alteration = 0
        self.num_grasped = 0

        self.end2end_rl = config['end2end_rl']
        if self.end2end_rl:
            config['grip_ctrl'] = True
            self.use_graspnet = False
            self.use_collected_states = False
            self.grasp_filter = None
            self.use_grasp_filter = False
        else:
            config['grip_ctrl'] = False

        self.env = env_class(config)

        if not self.end2end_rl:
            self.use_graspnet = config['use_graspnet']
            if self.use_graspnet:
                grasp_estimator_module = importlib.import_module(graspnet_module_name)
                self.grasp_estimator = grasp_estimator_module.GraspEstimator(cam_intrinsic=self.env.cam_intrinsic,
                                                                             cam_extrinsic=self.env.cam_extrinsic,
                                                                             factor_depth=1.0,
                                                                             img_height=self.env.render_height,
                                                                             img_width=self.env.render_width)
            self.use_collected_states = config['use_collected_states']
            if self.use_collected_states:
                self.last_data = None
                self.data_loader = DataLoader(path=config['sg_data_path'],
                                              shape=config['hook_shape'],
                                              start_dir_ind=config['sg_data_start_dir_ind'],
                                              end_dir_ind=config['sg_data_end_dir_ind'],
                                              end_file_ind=config['sg_data_end_file_ind'],
                                              num_dp_per_file=10000, num_hooks=config['num_hooks'])
            self.grasp_filter = grasp_filter
            if self.grasp_filter is not None:
                self.use_grasp_filter = True
            else:
                self.use_grasp_filter = False

        self.action_space = self.env.action_space
        self.step_count = 0

    def seed(self, seed=0):
        return self.env.seed(seed)

    def step(self, action):
        if isinstance(action, list):
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        self.scene_alteration += info['not_grasped_hook_movement']
        self.distance += info['distance']
        if self.end2end_rl and info['is_grasped']:
            self.num_grasped += 1
        if done:
            terminal_conditions = [True,
                                   info['is_separated'],
                                   info['is_object_dropped'],
                                   info['is_exceeded_workspace'],
                                   info['is_lift_failed']]
            self.num_dones[terminal_conditions] += 1
            self.num_ep_timesteps += info['current_timesteps']

        return obs, reward, done, info

    def reset(self, repeat_last_state=False):
        if self.end2end_rl:
            # let the agnet learn to grasp and manipulate
            self.env._reset_sim(return_real_depth=False)
        else:
            if self.use_collected_states:
                # recorded grasping states
                mjc_state = self.last_data
                loaded = False
                while not loaded:
                    if (not repeat_last_state) or (mjc_state is None):
                        grasp, feature, mjc_state = self.data_loader.get_next_good_state()

                        if self.use_grasp_filter:
                            is_separated, is_dropped, scene_alteration_scores = \
                                self.grasp_filter.evaluate([grasp], [feature])
                            if self.grasp_filter.condition == 0:
                                # separation condition
                                if is_separated > self.grasp_filter.threshold_1:
                                    loaded = True
                            elif self.grasp_filter.condition == 1:
                                # dropping condition
                                if is_dropped < self.grasp_filter.threshold_2:
                                    loaded = True
                            elif self.grasp_filter.condition == 2:
                                # non-grasped object movement condition
                                if scene_alteration_scores < self.grasp_filter.threshold_3:
                                    loaded = True
                            else:
                                # separation and non-grasped object movement condition
                                if is_separated > self.grasp_filter.threshold_1 and scene_alteration_scores < self.grasp_filter.threshold_3:
                                    loaded = True
                            self.num_discarded_grasps += 1
                        else:
                            loaded = True
                self.last_data = mjc_state
                self.env.set_sim_state(mjc_state)
            else:
                # generate grasps on the fly
                success_grasp = False
                while not success_grasp:
                    pregrasp_mjc_poses, gg_mjc_poses = np.array([None]), np.array([None])
                    if self.use_graspnet:
                        enough_grasps = False
                        while not enough_grasps:
                            # reset and generate grasps until enough grasps are generated
                            _, real_depth = self.env._reset_sim(return_real_depth=True)
                            enough_grasps_1, pregrasp_mjc_poses, gg_mjc_poses, _, _, _ = self.grasp_estimator.compute_grasp_pose_for_mjc_data(
                                real_depth=real_depth[0],
                                graspnet_to_mjc_gg_mat=self.env.graspnet_to_mjc_gg_mat,
                                mjc_gg_to_pregrasp_mat=self.env.gg_to_pregrasp_mat,
                                cam_name=self.env.graspnet_render_cams[0],
                                num_grasps=5,
                                visualisation=False)
                            enough_grasps_2, pregrasp_mjc_poses_2, gg_mjc_poses_2, _, _, _ = self.grasp_estimator.compute_grasp_pose_for_mjc_data(
                                real_depth=real_depth[1],
                                graspnet_to_mjc_gg_mat=self.env.graspnet_to_mjc_gg_mat,
                                mjc_gg_to_pregrasp_mat=self.env.gg_to_pregrasp_mat,
                                num_grasps=5,
                                cam_name=self.env.graspnet_render_cams[1],
                                visualisation=False)

                            if enough_grasps_1:
                                enough_grasps = True
                                if enough_grasps_2:
                                    pregrasp_mjc_poses = np.concatenate([pregrasp_mjc_poses, pregrasp_mjc_poses_2])
                                    gg_mjc_poses = np.concatenate([gg_mjc_poses, gg_mjc_poses_2])
                            else:
                                if enough_grasps_2:
                                    enough_grasps = True
                                    pregrasp_mjc_poses = pregrasp_mjc_poses_2
                                    gg_mjc_poses = gg_mjc_poses_2
                    else:
                        self.env._reset_sim(return_real_depth=False)

                    for i in range(pregrasp_mjc_poses.shape[0]):
                        success_grasp, _ = self.env._set_eef_start_pose(pregrasp_pose=pregrasp_mjc_poses[i],
                                                                        gg_pose=gg_mjc_poses[i])
                        if success_grasp:
                            break

        return self.env.reset()

    def render(self, mode='rgb_array', width=None, height=None, cam=None, segmentation=False):
        if mode == 'human':
            return self.env.render()
        else:
            return self.env.render(mode=mode, width=width, height=height, cam=cam, segmentation=segmentation)

    def close(self):
        self.env.close()


class MpVecEnvWrapper(object):
    def __init__(self, sg_data_path, config, num_envs=5, grasp_filter=None,
                 env_module_name='manipulation_project.env.entangled_env'):
        self.waiting = False
        self.closed = False
        self.num_env = num_envs
        self.image_obs = config['image_obs']
        self.num_passed_grasps = 0
        self.num_discarded_grasps = 0
        self.num_dones = np.zeros(shape=(5,))
        self.num_ep_timesteps = 0
        self.distance = 0
        self.scene_alteration = 0
        self.num_grasped = 0
        self.last_data = None

        self.end2end_rl = config['end2end_rl']
        if self.end2end_rl:
            config['grip_ctrl'] = True
            self.use_graspnet = False
            self.use_collected_states = False
            self.grasp_filter = None
            self.use_grasp_filter = False
        else:
            config['grip_ctrl'] = False

        forkserver_available = "forkserver" in mp.get_all_start_methods()
        mp_start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(mp_start_method)
        self.env_ids = [_ for _ in range(self.num_env)]
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_env)])
        self.processes = []

        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, env_module_name, config)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_action_dim", None))
        num_action = self.remotes[0].recv()
        self.action_space = spaces.multi_discrete.MultiDiscrete(nvec=[num_action for _ in range(self.num_env)])

        if not self.end2end_rl:
            self.use_graspnet = config['use_graspnet']
            if self.use_graspnet:
                graspnet_module = importlib.import_module(graspnet_module_name)
                self.remotes[0].send(("get_cam_params", None))
                cam_intrinsic, cam_extrinsic, render_width, render_height = self.remotes[0].recv()
                self.grasp_estimator = graspnet_module.GraspEstimator(cam_intrinsic=cam_intrinsic,
                                                                      cam_extrinsic=cam_extrinsic,
                                                                      factor_depth=1.0, img_height=render_height,
                                                                      img_width=render_width)
                self.remotes[0].send(("get_gg_to_eef_mat", None))
                self.graspnet_render_cams, self.graspnet_to_mjc_gg_mat, self.gg_to_pregrasp_mat = self.remotes[0].recv()
            self.use_collected_states = config['use_collected_states']
            if self.use_collected_states:
                self.data_loader = DataLoader(path=sg_data_path,
                                              shape=config['hook_shape'],
                                              start_dir_ind=config['sg_data_start_dir_ind'],
                                              end_dir_ind=config['sg_data_end_dir_ind'],
                                              end_file_ind=config['sg_data_end_file_ind'],
                                              num_dp_per_file=10000, num_hooks=config['num_hooks'])

            self.grasp_filter = grasp_filter
            if self.grasp_filter is not None:
                self.use_grasp_filter = True
            else:
                self.use_grasp_filter = False

        self.step_count = 0

    def step(self, actions):
        for env_id in self.env_ids:
            self.remotes[env_id].send(("step", actions[env_id]))

        results = [self.remotes[env_id].recv() for env_id in self.env_ids]
        obs, rewards, dones, infos = zip(*results)
        obs = list(obs)
        for env_id in self.env_ids:
            self.scene_alteration += infos[env_id]['not_grasped_hook_movement']
            self.distance += infos[env_id]['distance']
            if self.end2end_rl and infos[env_id]['is_grasped']:
                self.num_grasped += 1

            if dones[env_id]:
                infos[env_id]["terminal_observation"] = obs[env_id].copy()
                terminal_conditions = [True,
                                       infos[env_id]['is_separated'],
                                       infos[env_id]['is_object_dropped'],
                                       infos[env_id]['is_exceeded_workspace'],
                                       infos[env_id]['is_lift_failed']]
                self.num_dones[terminal_conditions] += 1
                self.num_ep_timesteps += infos[env_id]['current_timesteps']

                obs[env_id] = self.reset_indv(env_id)

        self.step_count += self.num_env

        return np.asarray(obs), np.asarray(rewards), np.asarray(dones), infos

    def seed(self, seed):
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        # Init loggers
        self.num_passed_grasps = 0
        self.num_discarded_grasps = 0
        self.num_dones = np.zeros(shape=(5,))
        self.num_ep_timesteps = 0
        self.distance = 0
        self.scene_alteration = 0
        self.step_count = 0

        obs = []
        for env_id in self.env_ids:
            indv_obs = self.reset_indv(env_id)
            obs.append(indv_obs)
        return np.asarray(obs)

    def render(self):
        self.remotes[0].send(("render", 'human'))
        _ = self.remotes[0].recv()

    def reset_indv(self, env_id):
        if self.end2end_rl:
            self.remotes[env_id].send(("reset_sim", False))
            _ = self.remotes[env_id].recv()
        else:
            if self.use_collected_states:
                loaded = False
                while not loaded:
                    grasp, feature, mjc_state = self.data_loader.get_next_good_state()
                    self.num_passed_grasps += 1

                    if self.use_grasp_filter:
                        is_separated, is_dropped, scene_alteration_scores = \
                            self.grasp_filter.evaluate([grasp], [feature])
                        if self.grasp_filter.condition == 0:
                            # separation condition
                            if is_separated > self.grasp_filter.threshold_1:
                                loaded = True
                            else:
                                self.num_discarded_grasps += 1
                        elif self.grasp_filter.condition == 1:
                            # dropping condition
                            if is_dropped < self.grasp_filter.threshold_2:
                                loaded = True
                            else:
                                self.num_discarded_grasps += 1
                        elif self.grasp_filter.condition == 2:
                            # non-grasped object movement condition
                            if scene_alteration_scores < self.grasp_filter.threshold_3:
                                loaded = True
                            else:
                                self.num_discarded_grasps += 1
                        else:
                            # separation and non-grasped object movement condition
                            if is_separated > self.grasp_filter.threshold_1 and scene_alteration_scores < self.grasp_filter.threshold_3:
                                loaded = True
                            else:
                                self.num_discarded_grasps += 1
                    else:
                        loaded = True

                self.remotes[env_id].send(("set_sim_state", mjc_state))
                self.remotes[env_id].recv()
            else:
                success_grasp, obs = False, None
                while not success_grasp:
                    pregrasp_mjc_poses, gg_mjc_poses = np.array([None]), np.array([None])
                    if self.use_graspnet:
                        enough_grasps = False
                        while not enough_grasps:
                            self.remotes[env_id].send(("reset_sim", True))
                            real_depth = self.remotes[env_id].recv()
                            enough_grasps_1, pregrasp_mjc_poses, gg_mjc_poses, _, _, _ = self.grasp_estimator.compute_grasp_pose_for_mjc_data(
                                real_depth=real_depth[0],
                                graspnet_to_mjc_gg_mat=self.graspnet_to_mjc_gg_mat,
                                mjc_gg_to_pregrasp_mat=self.gg_to_pregrasp_mat,
                                cam_name=self.graspnet_render_cams[0],
                                num_grasps=5,
                                visualisation=False)
                            enough_grasps_2, pregrasp_mjc_poses_2, gg_mjc_poses_2, _, _, _ = self.grasp_estimator.compute_grasp_pose_for_mjc_data(
                                real_depth=real_depth[1],
                                graspnet_to_mjc_gg_mat=self.graspnet_to_mjc_gg_mat,
                                mjc_gg_to_pregrasp_mat=self.gg_to_pregrasp_mat,
                                num_grasps=5,
                                cam_name=self.graspnet_render_cams[1],
                                visualisation=False)

                            if enough_grasps_1:
                                enough_grasps = True
                                if enough_grasps_2:
                                    pregrasp_mjc_poses = np.concatenate([pregrasp_mjc_poses, pregrasp_mjc_poses_2])
                                    gg_mjc_poses = np.concatenate([gg_mjc_poses, gg_mjc_poses_2])
                            else:
                                if enough_grasps_2:
                                    enough_grasps = True
                                    pregrasp_mjc_poses = pregrasp_mjc_poses_2
                                    gg_mjc_poses = gg_mjc_poses_2
                    else:
                        self.remotes[env_id].send(("reset_sim", False))
                        _ = self.remotes[env_id].recv()

                    for i in range(pregrasp_mjc_poses.shape[0]):
                        self.remotes[env_id].send(("set_grasp", (pregrasp_mjc_poses[i], gg_mjc_poses[i])))
                        success_grasp = self.remotes[env_id].recv()
                        if success_grasp:
                            break

        self.remotes[env_id].send(("reset", False))
        obs = self.remotes[env_id].recv()
        return np.asarray(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


def _worker(remote, parent_remote, env_module_name, env_config):
    parent_remote.close()
    env_module = importlib.import_module(env_module_name)
    env = env_module.EntangledEnv(config=env_config)

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset_sim":
                real_depth = []
                did_reset_sim = False
                while not did_reset_sim:
                    did_reset_sim, real_depth = env._reset_sim(return_real_depth=data)
                remote.send(real_depth)
            elif cmd == "set_grasp":
                success_grasp, _ = env._set_eef_start_pose(pregrasp_pose=data[0],
                                                           gg_pose=data[1])
                remote.send(success_grasp)
            elif cmd == "set_sim_state":
                env.set_sim_state(data)
                remote.send(None)
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_action_dim":
                remote.send(env.action_space.n)
            elif cmd == "get_cam_params":
                remote.send((env.cam_intrinsic, env.cam_extrinsic, env.render_width, env.render_height))
            elif cmd == "get_gg_to_eef_mat":
                remote.send((env.graspnet_render_cams, env.graspnet_to_mjc_gg_mat, env.gg_to_pregrasp_mat))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
