import numpy as np
from scipy.spatial.transform import Rotation
from manipulation_project.env.base_env import GraspingPrimitiveEnv
from manipulation_project.env.utils import mocap_set_action, ctrl_set_action, reset_mocap_welds, reset_mocap2body_xpos, \
    construct_transformation_matrix, construct_mjc_pose_from_transformation_matrix, zero_one_normalise
from copy import deepcopy as dcp
import matplotlib.pyplot as plt


class EntangledEnv(GraspingPrimitiveEnv):
    def __init__(self, config, debug=False):
        initial_qpos = {
            'gripper': [0.0, 0.0, 0.8 + 0.109, 0, 1, 0, 0],
            'kuka': [2.960, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000]
        }

        try:
            hook_shape = config['hook_shape']
        except:
            hook_shape = 'C'
        model_path = 'kuka_scene_'+hook_shape+'.xml'
        self.num_hooks = config['num_hooks']
        self.hook_keypoint_name = ['centre_site', 'left_site', 'right_site']
        self.num_reset_hook_sim_steps = 50
        self.hook_displacement = [0.0, 0.0, -0.012]
        if hook_shape != 'C':
            self.hook_keypoint_name = ['centre_site', 'left_site', 'right_site', 'left_end_site', 'right_end_site']
            self.num_reset_hook_sim_steps = 80
            self.hook_displacement = [0.0, 0.0, -0.015]

        self.end2end_rl = config['end2end_rl']
        self.coef_r_distance = config['coef_r_distance']
        self.coef_r_scene_alteration = config['coef_r_scene_alteration']

        self.image_obs = config['image_obs']
        self.depth = config['depth']
        self.image_obs_size = config['image_obs_size']
        self.view_1 = 'sideview1'
        self.view_2 = 'sideview2'

        self.hook_reset_pose_offset = None
        self.hook_grip_pose_offset = None
        # rotate graspnet output gripper tip frame conversion into the mujoco gripper one
        y_rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        z_rot = Rotation.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()
        rot = y_rot @ z_rot
        self.graspnet_to_mjc_gg_mat = construct_transformation_matrix([0.0, 0.0, 0.01], rot)
        # transformation from the gripper tip to the pregrasp pose
        gg_to_pregrasp_rot_mat = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        gg_to_pregrasp_mat = construct_transformation_matrix([0.0, 0.0, -0.05], gg_to_pregrasp_rot_mat)
        self.gg_to_pregrasp_mat = self.graspnet_to_mjc_gg_mat @ gg_to_pregrasp_mat

        y_rot_ = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
        rot_hook_to_gg = z_rot @ y_rot_
        hook_to_gg_mat = construct_transformation_matrix([0.0, 0.0, 0.0], rot_hook_to_gg)
        self.hook_to_pregrasp_mat = hook_to_gg_mat @ gg_to_pregrasp_mat

        self.gripper_finger_geom_names = ['l_fingertip_g0', 'r_fingertip_g0']
        self.grip_finger_full_open_ctrl = -0.02
        self.grip_finger_pregrasp_ctrl = 0.003
        self.grip_finger_close_ctrl = 0.07
        self.grasped_hook_id = 1
        self.hook_last_pos = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(self.num_hooks)]
        self.stable_grasp_test_actions = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        self.workspace_bounding_box = [-0.15, 0.15, -0.15, 0.15, 0.38, 0.7]  # -x, +x, -y, +y, -z, +z
        super().__init__(model_path, initial_qpos,
                         primitive=config['primitive'], grip_ctrl=config['grip_ctrl'],
                         view_1=self.view_1, view_2=self.view_2,
                         n_substeps=20, episode_length=config['episode_length'], debug=debug, seed=config['seed'])

    def set_sim_state(self, state):
        super().set_sim_state(state)
        self.sim.data.ctrl[:] = [self.grip_finger_close_ctrl, -self.grip_finger_close_ctrl]
        self._init_episode_data()

    def _reset_task(self):
        num_hooks = self.num_hooks
        # generate two entangled hook objects at random pos/ori
        hook_pose = np.ones((7,))
        rot_1 = self.np_random.uniform(low=-3.13, high=3.13, size=(3,))
        # rot_1[1] *= 0
        rot_1 = Rotation.from_euler('xyz', rot_1)
        quat_1 = rot_1.as_quat()
        quat_1 = np.concatenate([[quat_1[-1]], quat_1[:-1]])
        xpos_1 = np.array([0.0, 0.0, 0.5])
        hook_pose[:3] = xpos_1
        hook_pose[3:] = quat_1
        self.sim.data.set_joint_qpos('hook1_joint', hook_pose)

        hook_pose_2 = np.ones((7,))
        delta_rot = Rotation.from_euler('xyz', [180, 0, self.np_random.uniform(-1.0, 1.0) * 90], degrees=True)
        rot_2 = rot_1 * delta_rot
        quat_2 = rot_2.as_quat()
        quat_2 = np.concatenate([[quat_2[-1]], quat_2[:-1]])
        hook_pose_2[:3] = hook_pose[:3] + rot_2.apply(self.hook_displacement)
        hook_pose_2[3:] = quat_2
        self.sim.data.set_joint_qpos('hook2_joint', hook_pose_2)

        if num_hooks > 2:
            hook_pose_3 = np.ones((7,))
            delta_rot = Rotation.from_euler('xyz', [90, 0, self.np_random.uniform(-1.0, 1.0) * 90], degrees=True)
            rot_3 = rot_1 * delta_rot
            quat_3 = rot_3.as_quat()
            quat_3 = np.concatenate([[quat_3[-1]], quat_3[:-1]])
            hook_pose_3[:3] = hook_pose[:3] + rot_3.apply(self.hook_displacement)
            hook_pose_3[3:] = quat_3
            self.sim.data.set_joint_qpos('hook3_joint', hook_pose_3)
        else:
            self.sim.data.set_joint_qpos('hook3_joint', [-0.5, -0.1, 0, 1, 0, 0, 0])

        if num_hooks > 3:
            hook_pose_4 = np.ones((7,))
            delta_rot = Rotation.from_euler('xyz', [0, 90, self.np_random.uniform(-1.0, 1.0) * 90], degrees=True)
            rot_4 = rot_1 * delta_rot
            quat_4 = rot_4.as_quat()
            quat_4 = np.concatenate([[quat_4[-1]], quat_4[:-1]])
            hook_pose_4[:3] = hook_pose[:3] + rot_4.apply(self.hook_displacement)
            hook_pose_4[3:] = quat_4
            self.sim.data.set_joint_qpos('hook4_joint', hook_pose_4)
        else:
            self.sim.data.set_joint_qpos('hook4_joint', [-0.5, 0.1, 0, 1, 0, 0, 0])

        self.sim.forward()
        self.sim.data.ctrl[:] = [self.grip_finger_pregrasp_ctrl, -self.grip_finger_pregrasp_ctrl]
        # self.render()
        for _ in range(self.num_reset_hook_sim_steps):
            # rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size, height=self.image_obs_size)
            # plt.imshow(rgb_1.copy()[::-1, :, :])
            # plt.pause(0.00001)
            # self.render()
            self.sim.step()
        # plt.close()
        self.reset_task_state = dcp(self.sim.get_state())

    def _set_eef_start_pose(self, pregrasp_pose=None, gg_pose=None):
        self.sim.set_state(self.reset_task_state)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        self.sim.forward()
        self.sim.data.ctrl[:] = [self.grip_finger_pregrasp_ctrl, -self.grip_finger_pregrasp_ctrl]

        if gg_pose is None:
            n = self.np_random.integers(low=2) + 1
            pos = self.sim.data.get_site_xpos('hook1_' + self.hook_keypoint_name[n] + '_grip')
            quat = self.sim.data.get_body_xquat('hook1')
            hook_rot = Rotation.from_quat(np.concatenate([quat[1:], [quat[0]]])).as_matrix()
            hook_mat = construct_transformation_matrix(pos, hook_rot)
            pregrasp_mat = hook_mat @ self.hook_to_pregrasp_mat
            pregrasp_pose = construct_mjc_pose_from_transformation_matrix(pregrasp_mat)
            gg_pose = np.concatenate([pos, pregrasp_pose[3:]])

        mocap_set_action(self.sim, pregrasp_pose, reset_mocap_pos=False, delta=False)
        for _ in range(6):
            # rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size, height=self.image_obs_size)
            # plt.imshow(rgb_1.copy()[::-1, :, :])
            # plt.pause(0.00001)
            # self.render()
            self.sim.step()
        mocap_set_action(self.sim, gg_pose, reset_mocap_pos=False, delta=False)
        for _ in range(6):
            # rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size, height=self.image_obs_size)
            # plt.imshow(rgb_1.copy()[::-1, :, :])
            # plt.pause(0.00001)
            # self.render()
            self.sim.step()

        self.sim.data.ctrl[:] = [self.grip_finger_close_ctrl, -self.grip_finger_close_ctrl]
        for _ in range(10):
            # rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size, height=self.image_obs_size)
            # plt.imshow(rgb_1.copy()[::-1, :, :])
            # plt.pause(0.00001)
            # self.render()
            self.sim.step()
        # plt.close()
        l_finger_tip_xpos = self.sim.data.get_site_xpos('l_finger_tip_site')
        r_finger_tip_xpos = self.sim.data.get_site_xpos('r_finger_tip_site')
        finger_width = np.linalg.norm(l_finger_tip_xpos - r_finger_tip_xpos, axis=-1)
        successful_grasp = (finger_width > 0.0163) and (finger_width < 0.036)
        # print("Grasp ok:", successful_grasp, finger_width)
        grasped_state = dcp(self.sim.get_state())
        if successful_grasp:
            for a in self.stable_grasp_test_actions:
                self._set_action(a)
            #     rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size, height=self.image_obs_size)
            #     plt.imshow(rgb_1.copy()[::-1, :, :])
            #     plt.pause(0.00001)
            # plt.close()

            l_finger_tip_xpos = self.sim.data.get_site_xpos('l_finger_tip_site')
            r_finger_tip_xpos = self.sim.data.get_site_xpos('r_finger_tip_site')
            finger_width = np.linalg.norm(l_finger_tip_xpos - r_finger_tip_xpos, axis=-1)
            successful_lift = (finger_width > 0.0163) and (finger_width < 0.036)
            # print("Lift ok:", successful_lift, finger_width)

            if successful_lift:
                self.sim.set_state(grasped_state)
                reset_mocap_welds(self.sim)
                reset_mocap2body_xpos(self.sim)
                self.sim.forward()
                self._init_episode_data()
                if self.grasped_hook_id != -1:
                    return True, grasped_state

        return False, grasped_state

    def _init_episode_data(self):
        self.grasped_hook_id = self.get_grasped_hook_id()
        for n in range(self.num_hooks):
            self.hook_last_pos[n] = np.concatenate(
                (self.sim.data.get_site_xpos('hook' + str(n + 1) + '_centre_site').copy(),
                 self.sim.data.get_body_xquat('hook' + str(n + 1)).copy()))

    def get_grasped_hook_id(self):
        hook_num_contacts = [0 for _ in range(self.num_hooks)]
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(contact.geom1) in self.gripper_finger_geom_names:
                if self.sim.model.geom_id2name(contact.geom2) is not None:
                    if 'hook1' in self.sim.model.geom_id2name(contact.geom2):
                        hook_num_contacts[0] += 1
                    elif 'hook2' in self.sim.model.geom_id2name(contact.geom2):
                        hook_num_contacts[1] += 1
                    elif 'hook3' in self.sim.model.geom_id2name(contact.geom2):
                        hook_num_contacts[2] += 1
                    elif 'hook4' in self.sim.model.geom_id2name(contact.geom2):
                        hook_num_contacts[3] += 1
                    else:
                        continue
        if sum(hook_num_contacts) == 0:
            return -1
        return np.argmax(hook_num_contacts) + 1

    def _get_obs(self):
        hook_state = []
        hook_pose = []
        for n in range(self.num_hooks):
            tmp_hook_state = []
            for name in self.hook_keypoint_name:
                tmp_hook_state.append(self.sim.data.get_site_xpos('hook' + str(n + 1) + '_' + name))
            tmp_hook_state.append(self.sim.data.get_body_xquat('hook' + str(n + 1)))
            hook_state.append(np.concatenate(tmp_hook_state))
            hook_pose.append(np.concatenate((tmp_hook_state[0].copy(), tmp_hook_state[-1].copy())))

        gripper_tip_xpos = self.sim.data.get_site_xpos('grip_site')
        gripper_tip_xquat = self.sim.data.get_body_xquat('eef')
        l_finger_tip_xpos = self.sim.data.get_site_xpos('l_finger_tip_site')
        r_finger_tip_xpos = self.sim.data.get_site_xpos('r_finger_tip_site')
        finger_width = np.linalg.norm(l_finger_tip_xpos - r_finger_tip_xpos, axis=-1)
        if self.end2end_rl:
            gripper_state = [gripper_tip_xpos, gripper_tip_xquat, [finger_width]]
        else:
            gripper_state = [gripper_tip_xpos, gripper_tip_xquat, [finger_width, self.grasped_hook_id]]

        reward, is_terminate, info = self._compute_reward(hook_pose, gripper_tip_xpos, finger_width)
        self.hook_last_pos = dcp(hook_pose)

        state = np.concatenate(gripper_state + hook_state)

        if self.image_obs:
            rgb_1, depth_1 = self.render(mode='rgb_array', cam=self.view_1, width=self.image_obs_size,
                                         height=self.image_obs_size)
            depth_1 = (zero_one_normalise(depth_1, low=0.9, high=1.01) * 255).astype(np.uint8)[::-1, :]
            rgb_2, depth_2 = self.render(mode='rgb_array', cam=self.view_2, width=self.image_obs_size,
                                         height=self.image_obs_size)
            depth_2 = (zero_one_normalise(depth_2, low=0.9, high=1.01) * 255).astype(np.uint8)[::-1, :]
            info['state'] = state.copy()
            state = np.concatenate([
                rgb_1.copy()[::-1, :, :],
                rgb_2.copy()[::-1, :, :],
            ], axis=2)
            if self.depth:
                info['depth'] = np.concatenate([
                    depth_1.reshape((self.image_obs_size, self.image_obs_size, 1)),
                    depth_2.reshape((self.image_obs_size, self.image_obs_size, 1))
                ], axis=2)

        return state, reward, info

    def _compute_reward(self, hook_pose, gripper_tip_xyz, finger_width):
        # reward design:
        #   1. object separation reward
        #   2. success when an object is picked up and the other remains on the table
        r_step = 0.0
        is_exceeded_workspace = (gripper_tip_xyz[0] <= self.workspace_bounding_box[0]) or \
                                (gripper_tip_xyz[0] >= self.workspace_bounding_box[1]) or \
                                (gripper_tip_xyz[1] <= self.workspace_bounding_box[2]) or \
                                (gripper_tip_xyz[1] >= self.workspace_bounding_box[3]) or \
                                (gripper_tip_xyz[2] >= self.workspace_bounding_box[-1])
        if is_exceeded_workspace:
            r_step = -10

        if not self.end2end_rl:
            grasped_hook_pose = hook_pose[self.grasped_hook_id - 1].copy()[:3]

            is_object_dropped = (finger_width < 0.0163) or (finger_width > 0.038)
            is_grasped = (finger_width > 0.0163) and (finger_width < 0.036)

            if is_object_dropped:
                r_step = -10

            is_terminate = is_exceeded_workspace or is_object_dropped

            is_separated = False
            is_lift_failed = False
            if (not is_terminate) and (self.ep_step_count == self.episode_length):
                is_separated = self._lift_test()
            if is_separated:
                r_step = 10
            else:
                is_lift_failed = True

            # rewarding agent for separating the grasped object
            distance = np.linalg.norm(grasped_hook_pose - np.array(hook_pose)[:, :3], axis=-1)
            distance = distance.sum() / (self.num_hooks - 1)
            r_distance = self.coef_r_distance * distance

            # punish agent for moving other objects
            not_grasped_hook_displacement = np.linalg.norm(np.array(hook_pose) - np.array(self.hook_last_pos), axis=-1)
            not_grasped_hook_displacement[self.grasped_hook_id - 1] = 0.0
            not_grasped_hook_displacement = not_grasped_hook_displacement.sum() / (self.num_hooks - 1)
            r_not_grasped_hook_displacement = self.coef_r_scene_alteration * not_grasped_hook_displacement
            r = r_distance + r_not_grasped_hook_displacement + r_step
        else:
            is_terminate = is_exceeded_workspace
            r_grasped = 0
            is_object_dropped = (finger_width < 0.0163) or (finger_width > 0.038)
            is_grasped = (finger_width > 0.0163) and (finger_width < 0.036)
            if is_grasped:
                self.grasped_hook_id = self.get_grasped_hook_id()
            if self.grasped_hook_id == -1:
                is_grasped = False

            if is_grasped:
                r_grasped = 5
                grasped_hook_pose = hook_pose[self.grasped_hook_id - 1].copy()[:3]
                # rewarding agent for separating the grasped object
                distance = np.linalg.norm(grasped_hook_pose - np.array(hook_pose)[:, :3], axis=-1)
                distance = distance.sum() / (self.num_hooks - 1)
                # punish agent for moving other objects
                not_grasped_hook_displacement = np.linalg.norm(np.array(hook_pose) - np.array(self.hook_last_pos), axis=-1)
                not_grasped_hook_displacement[self.grasped_hook_id - 1] = 0.0
                not_grasped_hook_displacement = not_grasped_hook_displacement.sum() / (self.num_hooks - 1)
            else:
                distance = 0.0
                not_grasped_hook_displacement = 0.0

            r_distance = self.coef_r_distance * distance
            r_not_grasped_hook_displacement = self.coef_r_scene_alteration * not_grasped_hook_displacement

            is_separated = False
            is_lift_failed = False
            if (not is_terminate) and (self.ep_step_count == self.episode_length) and is_grasped:
                is_separated = self._lift_test()
            if is_separated:
                r_step = 10
            else:
                is_lift_failed = True

            r = r_grasped + r_distance + r_not_grasped_hook_displacement + r_step

        return r, is_terminate, {'distance': distance,
                                 'not_grasped_hook_movement': not_grasped_hook_displacement,
                                 'r_step': r_step,
                                 'is_terminate': is_terminate,
                                 'is_separated': is_separated,
                                 'is_lift_failed': is_lift_failed,
                                 'is_object_dropped': is_object_dropped,
                                 'is_grasped': is_grasped,
                                 'is_exceeded_workspace': is_exceeded_workspace,
                                 'finger_width': finger_width,
                                 'current_timesteps': dcp(self.ep_step_count)}

    def _lift_test(self):
        prev_states = dcp(self.sim.get_state())

        for _ in range(2):
            # lift-up motions
            self._six_dof_primitive.set_action(4, self.sim)
            # self.render(mode="human")

        l_finger_tip_xpos = self.sim.data.get_site_xpos('l_finger_tip_site')
        r_finger_tip_xpos = self.sim.data.get_site_xpos('r_finger_tip_site')
        finger_width = np.linalg.norm(l_finger_tip_xpos - r_finger_tip_xpos, axis=-1)
        is_object_dropped = (finger_width < 0.0163) or (finger_width > 0.038)
        if is_object_dropped:
            self.sim.set_state(prev_states)
            reset_mocap_welds(self.sim)
            reset_mocap2body_xpos(self.sim)
            self.sim.forward()
            return False

        num_grasped_hook_contact_with_other_hooks = 0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if 'hook' + str(self.grasped_hook_id) in self.sim.model.geom_id2name(contact.geom1):
                if 'hook' in self.sim.model.geom_id2name(contact.geom2):
                    num_grasped_hook_contact_with_other_hooks += 1
            if 'hook' + str(self.grasped_hook_id) in self.sim.model.geom_id2name(contact.geom2):
                if 'hook' in self.sim.model.geom_id2name(contact.geom1):
                    num_grasped_hook_contact_with_other_hooks += 1
        self.sim.set_state(prev_states)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        self.sim.forward()

        if num_grasped_hook_contact_with_other_hooks > 0:
            return False
        else:
            return True

    def _env_setup(self, initial_qpos=None):
        for i in [1, 2, 3, 4, 5, 6, 7]:
            self.sim.data.set_joint_qpos('joint_' + str(i), initial_qpos['kuka'][i - 1])
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        self.sim.forward()
        mocap_set_action(self.sim, [0.0, 0.0, 0.7, 0, 0.707105, 0.707105, 0], reset_mocap_pos=False, delta=False)
        for _ in range(4):
            self.sim.step()

        ctrl_set_action(self.sim, [0, 0, 0, 0, 0, 0, 0, -5.0, 5.0])
        for _ in range(10):
            self.sim.step()
