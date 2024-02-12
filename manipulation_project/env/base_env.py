import os
import copy
import numpy as np
from typing import Optional

from gym import error, spaces, Env
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

from manipulation_project.env.utils import \
    reset_mocap_welds, reset_mocap2body_xpos, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
from manipulation_project.env.primitive_actions import SixDofMovements, HemisphereMovements, EmptyMovement


class GraspingPrimitiveEnv(Env):
    def __init__(self, model_path, initial_qpos, n_substeps, episode_length, primitive='HemisphereMovements', grip_ctrl=False,
                 debug=False, view_1='sideview', view_2='backview', seed=None):
        self.debug = debug
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'scene', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed(seed=seed)
        self.ep_step_count = 0
        self.episode_length = episode_length

        self.graspnet_render_cams = ['topview', 'frontview']
        self.view_1 = view_1
        self.view_2 = view_2
        self.render_width = 720
        self.render_height = 720

        self.initial_qpos = initial_qpos
        self._env_setup(initial_qpos=initial_qpos)
        self._viewer_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.cam_intrinsic, self.cam_extrinsic = self._get_cam_param(self.graspnet_render_cams)

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim, _ = self._reset_sim(return_real_depth=False)
        obs, _, _ = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        # primitive actions
        self._six_dof_primitive = SixDofMovements(translation_distance=0.05, rotation_degree=30, action_substep=10)
        if primitive == 'SixDofMovements':
            self.primitive = SixDofMovements(translation_distance=0.05, rotation_degree=30, action_substep=10, grip_ctrl=grip_ctrl)
        elif primitive == 'HemisphereMovements':
            self.primitive = HemisphereMovements(radius=0.035, action_substep=10, grip_ctrl=grip_ctrl)
        elif primitive == 'EmptyMovement':
            self.primitive = EmptyMovement()
        else:
            raise ValueError("Only support primitives: SixDofMovements & HemisphereMovements.")
        self.action_space = self.primitive.action_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._set_action(action)
        self.ep_step_count += 1
        obs, reward, info = self._get_obs()
        time_done = False

        if self.ep_step_count >= self.episode_length:
            time_done = True

        done = time_done or info['is_terminate']
        if self.debug:
            done = time_done

        return obs, reward, done, info

    def reset(
            self,
            *,
            eef_start_pose=None,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # call _reset_sim() to get a depth of the workspace, compute grasp start pose
        # call _reset_task() with an eef pose to setup the gripper
        # then call reset()
        self.ep_step_count = 0
        obs, _, _ = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=None, height=None, cam=None, segmentation=False):
        self._render_callback()
        if width is None:
            width = self.render_width
            height = self.render_height
        if mode == 'rgb_array':
            if cam is None:
                cam = self.view_1

            cam_id = self.sim.model.camera_name2id(cam)
            self._get_viewer(mode).render(width=width, height=height,
                                          camera_id=cam_id, segmentation=segmentation)
            data = self._get_viewer(mode).read_pixels(width=width,
                                                      height=height,
                                                      depth=True, segmentation=segmentation)
            return data[0], data[1]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def set_sim_state(self, state):
        self.sim.set_state(state)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        self.sim.forward()

    def _reset_sim(self, return_real_depth=False, return_color=False):
        """Resets a simulation and indicates whether it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt the reset again.
        """
        self.sim.set_state(self.initial_state)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        self.sim.forward()
        self._reset_task()
        # ctrl_set_action(self.sim, [0, 0, 0, 0, 0, 0, 0, -5.0, 5.0])
        # for _ in range(3):
        #     self.sim.step()
        if return_real_depth:
            color_1, depth_1 = self.render(mode='rgb_array', cam=self.graspnet_render_cams[0])
            real_depth_1 = get_real_depth_map(self.sim, depth_1).reshape((self.render_height, self.render_width))
            color_2, depth_2 = self.render(mode='rgb_array', cam=self.graspnet_render_cams[1])
            real_depth_2 = get_real_depth_map(self.sim, depth_2).reshape((self.render_height, self.render_width))
            real_depths = np.concatenate([[real_depth_1], [real_depth_2]])
            if return_color:
                return True, real_depths, color_1, color_2
            return True, real_depths
        return True, [None]

    def _reset_task(self):
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        self.primitive.set_action(action, self.sim)

    def _compute_reward(self, *args):
        raise NotImplementedError()

    def _get_cam_param(self, camera_names):
        intrinsics, extrinsics = dict(), dict()
        for camera_name in camera_names:
            cam_intrinsic = get_camera_intrinsic_matrix(sim=self.sim, camera_name=camera_name,
                                                        camera_height=self.render_height,
                                                        camera_width=self.render_width)
            cam_extrinsic = get_camera_extrinsic_matrix(sim=self.sim, camera_name=camera_name)
            camera_axis_correction = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )  # original_cam to corrected_cam frame transform
            camera_axis_correction_inv = np.linalg.inv(camera_axis_correction)  # corrected_cam to original_cam
            cam_extrinsic = cam_extrinsic @ camera_axis_correction_inv
            intrinsics[camera_name] = cam_intrinsic.copy()
            extrinsics[camera_name] = cam_extrinsic.copy()
        return intrinsics, extrinsics

    def _env_setup(self, initial_qpos=None):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass
