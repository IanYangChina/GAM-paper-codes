import os, math
import numpy as np
import mujoco_py
from manipulation_project.env.utils import *
from scipy.spatial.transform import Rotation


def reset_eef_pose(mjsim, pose):
    mjsim.data.set_joint_qpos('eef_joint', pose)
    reset_mocap_welds(mjsim)
    reset_mocap2body_xpos(mjsim)
    mjsim.forward()
    random_quat = Rotation.from_euler('xyz', np.random.uniform(-3.14, 3.14, (3,)), degrees=False).as_quat()
    random_quat = np.concatenate([[random_quat[-1]], random_quat[:-1]])
    mocap_ctrl = np.concatenate([[0.0, 0.0, 0.0], random_quat])
    mocap_set_action(mjsim, mocap_ctrl, reset_mocap_pos=False)
    for _ in range(5):
        mjsim.step()


model_path = "scene/kuka_scene.xml"
fullpath = os.path.join(os.path.dirname(__file__), '..', 'assets', model_path)
model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(model, nsubsteps=20)
viewer = mujoco_py.MjViewer(sim)
reset_mocap_welds(sim)
reset_mocap2body_xpos(sim)
sim.forward()
# 0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000

# init_rot = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()
# init_quat = np.concatenate([[init_rot[-1]], init_rot[:-1]]).tolist()
# reset_eef_pose(sim, [0.0, 0.0, 0.7+0.109]+init_quat)
# print(sim.data.get_mocap_pos('mocap'))

for _ in range(10000):
    # last_mocap_quat = sim.data.get_mocap_quat('mocap')
    # last_mocap_quat = np.concatenate([last_mocap_quat[1:], [last_mocap_quat[0]]])
    # last_mocap_rot = Rotation.from_quat(last_mocap_quat)
    # mocap_rot = last_mocap_rot * Rotation.from_euler('xyz', [10, 0, 0], degrees=True)
    # mocap_quat_ctrl = mocap_rot.as_quat()
    # mocap_quat_ctrl = np.concatenate([[mocap_quat_ctrl[-1]], mocap_quat_ctrl[:-1]])
    #
    # pos_ctrl, gripper_ctrl = np.array([0.0, 0.0, 0.0, 1.0])[:3], np.array([0.0, 0.0, 0.0, 1.0])[3]
    # pos_ctrl = mocap_rot.apply(pos_ctrl)
    # # pos_ctrl *= 0.05
    # gripper_ctrl = np.array([gripper_ctrl, -gripper_ctrl])
    # action = np.concatenate([pos_ctrl, mocap_quat_ctrl, gripper_ctrl])
    # ctrl_set_action(sim, action)
    # mocap_set_action(sim, action, reset_mocap_pos=False)
    sim.step()
    viewer.render()
