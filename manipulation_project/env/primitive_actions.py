import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation
from manipulation_project.env.utils import mocap_set_action


class EmptyMovement:
    def __init__(self):
        self._action_space = spaces.Discrete(n=1)

    @property
    def action_space(self):
        return self._action_space

    def set_action(self, action, sim):
        # Rotation - quat as [x, y, z, w]
        # Mujoco-py - quat as [w, x, y, z]
        last_mocap_quat = sim.data.get_mocap_quat('mocap')
        mocap_xpos_ctrl = [0.0, 0.0, 0.0]
        mocap_ctrl = np.concatenate([mocap_xpos_ctrl, last_mocap_quat])
        mocap_set_action(sim, mocap_ctrl, reset_mocap_pos=False)
        for _ in range(2):
            sim.step()


class SixDofMovements:
    """Primitives in gripper tip frame.
    0: move along +x direction for 0.02 meters
    1: move along -x direction for 0.02 meters
    2: move along +y direction for 0.02 meters
    3: move along -y direction for 0.02 meters
    4: move along +z direction for 0.02 meters
    5: move along -z direction for 0.02 meters
    6: rotate along x clockwise for 10 degree
    7: rotate along x counter-clockwise for 10 degree
    8: rotate along y clockwise for 10 degree
    9: rotate along y counter-clockwise for 10 degree
    10: rotate along z clockwise for 10 degree
    11: rotate along z counter-clockwise for 10 degree
    """
    def __init__(self, translation_distance, rotation_degree, action_substep, grip_ctrl=False):
        self.action_substep = action_substep
        self.translation_distance = translation_distance / self.action_substep
        self.rotation_degree = rotation_degree / self.action_substep

        self.mocap_delta_xpos = [
            [self.translation_distance, 0.0, 0.0],
            [-self.translation_distance, 0.0, 0.0],
            [0.0, self.translation_distance, 0.0],
            [0.0, -self.translation_distance, 0.0],
            [0.0, 0.0, self.translation_distance],
            [0.0, 0.0, -self.translation_distance],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        self.mocap_delta_rot = [
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [self.rotation_degree, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [-self.rotation_degree, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, self.rotation_degree, 0], degrees=True),
            Rotation.from_euler('xyz', [0, -self.rotation_degree, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, self.rotation_degree], degrees=True),
            Rotation.from_euler('xyz', [0, 0, -self.rotation_degree], degrees=True),
        ]
        self.n_mv_actions = len(self.mocap_delta_xpos)
        self.grip_ctrl = grip_ctrl
        self.grip_finger_full_open_ctrl = -0.02
        self.grip_finger_close_ctrl = 0.07
        if self.grip_ctrl:
            self._action_space = spaces.Discrete(n=self.n_mv_actions+2)
        else:
            self._action_space = spaces.Discrete(n=self.n_mv_actions)

    @property
    def action_space(self):
        return self._action_space

    def set_action(self, action, sim):
        if action < self.n_mv_actions:
            # Rotation - quat as [x, y, z, w]
            # Mujoco-py - quat as [w, x, y, z]
            for i in range(self.action_substep):
                last_mocap_quat = sim.data.get_mocap_quat('mocap')
                last_mocap_quat = np.concatenate([last_mocap_quat[1:], [last_mocap_quat[0]]])
                last_mocap_rot = Rotation.from_quat(last_mocap_quat)
                mocap_rot = last_mocap_rot * self.mocap_delta_rot[action]
                mocap_quat_ctrl = mocap_rot.as_quat()
                mocap_quat_ctrl = np.concatenate([[mocap_quat_ctrl[-1]], mocap_quat_ctrl[:-1]])

                mocap_xpos_ctrl = self.mocap_delta_xpos[action]
                mocap_ctrl = np.concatenate([mocap_xpos_ctrl, mocap_quat_ctrl])
                mocap_set_action(sim, mocap_ctrl, reset_mocap_pos=False)
                for _ in range(2):
                    sim.step()
        else:
            assert self.grip_ctrl, "grip_ctrl must be True"
            if action == self.n_mv_actions:
                sim.data.ctrl[:] = [self.grip_finger_full_open_ctrl, -self.grip_finger_full_open_ctrl]
            elif action == self.n_mv_actions + 1:
                sim.data.ctrl[:] = [self.grip_finger_close_ctrl, -self.grip_finger_close_ctrl]
            else:
                raise NotImplementedError()
            for _ in range(4):
                sim.step()


class HemisphereMovements:
    """Primitives that move a gripped object around a hemisphere
    https://link.springer.com/chapter/10.1007/978-3-662-62962-8_28
    """
    def __init__(self, radius, action_substep, grip_ctrl=False):
        self.action_substep = action_substep
        self.radius = radius
        self.radius_1 = 2 * np.sqrt(2) * self.radius / 3
        self.radius_2 = np.sqrt(5) * self.radius / 3
        self.mocap_delta_xpos = (np.array([
            # 0-7
            [self.radius, 0.0, 0.0],
            [self.radius / np.sqrt(2), self.radius / np.sqrt(2), 0.0],
            [0.0, self.radius, 0.0],
            [-self.radius / np.sqrt(2), self.radius / np.sqrt(2), 0.0],
            [-self.radius, 0.0, 0.0],
            [-self.radius / np.sqrt(2), -self.radius / np.sqrt(2), 0.0],
            [0.0, -self.radius, 0.0],
            [self.radius / np.sqrt(2), -self.radius / np.sqrt(2), 0.0],
            # 8-15
            [self.radius_1, 0.0, self.radius / 3],
            [self.radius_1 / np.sqrt(2), self.radius_1 / np.sqrt(2), self.radius / 3],
            [0.0, self.radius_1, self.radius / 3],
            [-self.radius_1 / np.sqrt(2), self.radius_1 / np.sqrt(2), self.radius / 3],
            [-self.radius_1, 0.0, self.radius / 3],
            [-self.radius_1 / np.sqrt(2), -self.radius_1 / np.sqrt(2), self.radius / 3],
            [0.0, -self.radius_1, self.radius / 3],
            [self.radius_1 / np.sqrt(2), -self.radius_1 / np.sqrt(2), self.radius / 3],
            # 16-23
            [self.radius_2, 0.0, 2 * self.radius / 3],
            [self.radius_2 / np.sqrt(2), self.radius_2 / np.sqrt(2), 2 * self.radius / 3],
            [0.0, self.radius_2, 2 * self.radius / 3],
            [-self.radius_2 / np.sqrt(2), self.radius_2 / np.sqrt(2), 2 * self.radius / 3],
            [-self.radius_2, 0.0, 2 * self.radius / 3],
            [-self.radius_2 / np.sqrt(2), -self.radius_2 / np.sqrt(2), 2 * self.radius / 3],
            [0.0, -self.radius_2, 2 * self.radius / 3],
            [self.radius_2 / np.sqrt(2), -self.radius_2 / np.sqrt(2), 2 * self.radius / 3],
            # 24
            [0.0, 0.0, self.radius]
        ]) / self.action_substep).tolist()
        rot_x_lv_2 = 19.47 / self.action_substep
        rot_x_lv_3 = 41.81 / self.action_substep
        self.mocap_delta_rot_in_world_frame = [
            # 0-7
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 45/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 90/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 135/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 180/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 225/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 270/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [0, 0, 315/self.action_substep], degrees=True),
            # 8-15
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 45/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 90/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 135/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 180/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 225/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 270/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_2, 0, 315/self.action_substep], degrees=True),
            # 16-23
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 0], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 45/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 90/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 135/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 180/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 225/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 270/self.action_substep], degrees=True),
            Rotation.from_euler('xyz', [rot_x_lv_3, 0, 315/self.action_substep], degrees=True),
            # 16-23
            Rotation.from_euler('xyz', [0, 0, 0], degrees=True),
        ]
        self.n_mv_actions = len(self.mocap_delta_xpos)
        self.grip_ctrl = grip_ctrl
        self.grip_finger_full_open_ctrl = -0.02
        self.grip_finger_close_ctrl = 0.07
        if self.grip_ctrl:
            self._action_space = spaces.Discrete(n=self.n_mv_actions+2)
        else:
            self._action_space = spaces.Discrete(n=self.n_mv_actions)

    @property
    def action_space(self):
        return self._action_space

    def set_action(self, action, sim):
        if action < self.n_mv_actions:
            # Rotation - quat as [x, y, z, w]
            # Mujoco-py - quat as [w, x, y, z]
            for i in range(self.action_substep):
                last_mocap_quat = sim.data.get_mocap_quat('mocap')
                last_mocap_quat = np.concatenate([last_mocap_quat[1:], [last_mocap_quat[0]]])
                last_mocap_rot = Rotation.from_quat(last_mocap_quat)
                mocap_rot = self.mocap_delta_rot_in_world_frame[action] * last_mocap_rot
                mocap_quat_ctrl = mocap_rot.as_quat()
                mocap_quat_ctrl = np.concatenate([[mocap_quat_ctrl[-1]], mocap_quat_ctrl[:-1]])

                mocap_xpos_ctrl = self.mocap_delta_xpos[action]
                mocap_ctrl = np.concatenate([mocap_xpos_ctrl, mocap_quat_ctrl])
                mocap_set_action(sim, mocap_ctrl, reset_mocap_pos=False)
                for _ in range(2):
                    sim.step()
        else:
            assert self.grip_ctrl, "grip_ctrl must be True"
            if action == self.n_mv_actions:
                sim.data.ctrl[:] = [self.grip_finger_full_open_ctrl, -self.grip_finger_full_open_ctrl]
            elif action == self.n_mv_actions + 1:
                sim.data.ctrl[:] = [self.grip_finger_close_ctrl, -self.grip_finger_close_ctrl]
            else:
                raise NotImplementedError()
            for _ in range(4):
                sim.step()
