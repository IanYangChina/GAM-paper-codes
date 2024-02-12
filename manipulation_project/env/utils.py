import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action, reset_mocap_pos=True, delta=True):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        if reset_mocap_pos:
            reset_mocap2body_xpos(sim)
        if delta:
            sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        else:
            sim.data.mocap_pos[:] = pos_delta
        sim.data.mocap_quat[:] = quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
        sim.model.eq_type is None
        or sim.model.eq_obj1id is None
        or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
        sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R


def get_real_depth_map(sim, depth_map):
    """
    By default, MuJoCo will return a depth map that is normalized in [0, 1]. This
    helper function converts the map so that the entries correspond to actual distances.

    (see https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L742)

    Args:
        sim (MjSim): simulator instance
        depth_map (np.array): depth map with values normalized in [0, 1] (default depth map
            returned by MuJoCo)
    Return:
        depth_map (np.array): depth map that corresponds to actual distances
    """
    # Make sure that depth values are normalized
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation

    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def construct_transformation_matrix(pos, rot_mat):
    t = np.array(
            [[*rot_mat[0], pos[0]],
             [*rot_mat[1], pos[1]],
             [*rot_mat[2], pos[2]],
             [0, 0, 0, 1]]
        )
    return t


def construct_mjc_pose_from_transformation_matrix(mat):
    xpos = mat[:-1, -1]
    rot_quat = Rotation.from_matrix(mat[:-1, :-1]).as_quat()
    rot_quat_mjc = np.concatenate([[rot_quat[-1]], rot_quat[:-1]])
    mjc_pose = np.concatenate([xpos, rot_quat_mjc])
    return mjc_pose


def construct_mjc_poses_from_transformation_matrices(mat):
    xpos = mat[:, :-1, -1]
    rot_quat = Rotation.from_matrix(mat[:, :-1, :-1]).as_quat()
    rot_quat_mjc = np.concatenate([np.transpose([rot_quat[:, -1]]), rot_quat[:, :-1]], axis=1)
    mjc_poses = np.concatenate([xpos, rot_quat_mjc], axis=1)
    return mjc_poses


def zero_one_normalise(x, low, high):
    return 1 + ((high - x) / (high - low))
