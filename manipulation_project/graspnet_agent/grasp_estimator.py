"""Codes adapted from https://github.com/graspnet/graspnet-baseline/blob/main/demo.py"""
import os
import torch
import open3d as o3d
import numpy as np
from PIL import Image
from graspnetAPI import GraspGroup
from manipulation_project.graspnet_agent.graspnet.graspnet import GraspNet, pred_decode
from manipulation_project.graspnet_agent.utils.collision_detector import ModelFreeCollisionDetector
from manipulation_project.graspnet_agent.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from manipulation_project.graspnet_agent.graspnet_checkpoints import checkpoint_dir
from manipulation_project.env.utils import construct_mjc_poses_from_transformation_matrices
from scipy.spatial.transform import Rotation
from copy import deepcopy


class GraspEstimator(object):
    def __init__(self,
                 cam_intrinsic, cam_extrinsic,
                 factor_depth=1.0, img_height=720, img_width=1280,
                 cuda_device_id=0):
        self.model = GraspNet(input_feature_dim=0,
                              num_view=300,
                              num_angle=12,
                              num_depth=4,
                              cylinder_radius=0.05,
                              hmin=-0.02,
                              hmax_list=[0.01, 0.02, 0.03, 0.04],
                              is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cuda_device_id != 0:
            self.device = torch.device("cuda:%i" % cuda_device_id)

        self.model.to(self.device)
        self.data_path = checkpoint_dir
        self.checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint_kn.tar"))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.start_epoch = self.checkpoint['epoch']
        self.model.eval()
        # for scene point cloud pre-processing
        self.img_height = img_height
        self.img_width = img_width
        self.num_point = int(20000)
        self.factor_depth = factor_depth

        self.cam_info = {
            'topview': self.get_cam_dict('topview', cam_intrinsic['topview'], cam_extrinsic['topview']),
            'frontview': self.get_cam_dict('frontview', cam_intrinsic['frontview'], cam_extrinsic['frontview'])
        }

        # for mujoco env
        self.mujoco_cam_xy_plane_mirror_correction_mat = np.linalg.inv(np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]
        ]))

        # for collision avoidance detection
        self.voxel_size = 0.01
        self.collision_thresh = 0.005

    def get_cam_dict(self, cam_name, intrinsic, extrinsic):
        return {'extrinsic': extrinsic,
                'workspace_mask': np.array(Image.open(os.path.join(self.data_path, cam_name + '_workspace_mask.png'))),
                'camera': CameraInfo(self.img_width, self.img_height,
                                     intrinsic[0][0], intrinsic[1][1],
                                     intrinsic[0][2], intrinsic[1][2], self.factor_depth)}

    # todo: compute grasps from real-world images
    def compute_grasp_pose_for_realworld_data(self, depth,
                                              color=None,
                                              cam_name='realsense',
                                              num_grasps=5,
                                              visualisation=False,
                                              vis_transform_into_world=False):
        gg, vp_features = self.predict(color=color, depth=depth, cam_name=cam_name,
                                       num_grasps=num_grasps,
                                       visualisation=visualisation,
                                       vis_transform_into_world=vis_transform_into_world,
                                       use_ws_mask=True)
        return

    def compute_grasp_pose_for_mjc_data(self, real_depth,
                                        graspnet_to_mjc_gg_mat, mjc_gg_to_pregrasp_mat,
                                        cam_name='topview',
                                        color=None, num_grasps=1,
                                        visualisation=False,
                                        vis_transform_into_world=False):
        gg, vp_features = self.predict(color=color, depth=real_depth, cam_name=cam_name,
                                       num_grasps=num_grasps,
                                       visualisation=visualisation,
                                       vis_transform_into_world=vis_transform_into_world,
                                       use_ws_mask=True)
        if num_grasps is not None:
            if gg.rotation_matrices.shape[0] < num_grasps:
                return False, [None], [None], [None], [None], [None]
        else:
            num_grasps = gg.rotation_matrices.shape[0]
            if num_grasps < 1:
                return False, [None], [None], [None], [None], [None]
        # transform graspnet outputs into the wierd mujoco camera conversion
        gg_transforms = []
        for n in range(num_grasps):
            gg_transform = np.array([
                [*gg.rotation_matrices[n][0], gg.translations[n][0]],
                [*gg.rotation_matrices[n][1], gg.translations[n][1]],
                [*gg.rotation_matrices[n][2], gg.translations[n][2]],
                [0, 0, 0, 1]
            ])
            # the output of the graspnet model is an x-y plane mirror of the original mujoco camera
            gg_transform_in_mjc_cam = self.mujoco_cam_xy_plane_mirror_correction_mat @ \
                                      gg_transform @ \
                                      self.mujoco_cam_xy_plane_mirror_correction_mat
            gg_transforms.append(gg_transform_in_mjc_cam)

        mjc_gg_transforms = np.asarray(gg_transforms) @ graspnet_to_mjc_gg_mat
        world_to_gg_transforms = self.cam_info[cam_name]['extrinsic'] @ mjc_gg_transforms
        gg_mjc_poses = construct_mjc_poses_from_transformation_matrices(world_to_gg_transforms)

        pregrasp_transforms = np.asarray(gg_transforms) @ mjc_gg_to_pregrasp_mat
        world_to_pregrasp_transforms = self.cam_info[cam_name]['extrinsic'] @ pregrasp_transforms
        pregrasp_mjc_poses = construct_mjc_poses_from_transformation_matrices(world_to_pregrasp_transforms)
        return True, pregrasp_mjc_poses, gg_mjc_poses, gg.grasp_group_array[:, -1].astype(np.int64), gg.grasp_group_array[:, 4:16], vp_features

    def predict(self, color, depth, cam_name='topview', num_grasps=5, visualisation=False,
                vis_transform_into_world=False,
                use_ws_mask=True):
        end_points, cloud = self.get_and_process_data(color, depth, use_ws_mask=use_ws_mask, cam_name=cam_name)
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg, collision_mask = self.collision_detection(gg, np.array(cloud.points))
        gg.nms()
        gg.sort_by_score()
        if num_grasps is not None:
            gg = gg[:num_grasps]
        if visualisation:
            self.vis_grasps(gg, cloud, transform_into_world=vis_transform_into_world, cam_name=cam_name)
        return gg, end_points['fp2_features'][0].detach().cpu().numpy()

    def get_and_process_data(self, color=None, depth=None, use_ws_mask=True, cam_name='topview'):
        if color is not None:
            color_tmp = color.copy()
            if color.dtype.name == "uint8":
                color_tmp = color_tmp / 255.0
            depth_tmp = depth.copy()
            # generate cloud
            cloud = create_point_cloud_from_depth_image(depth_tmp, self.cam_info[cam_name]['camera'], organized=True)
            mask = (self.cam_info[cam_name]['workspace_mask'] & (depth_tmp > 0))
            if not use_ws_mask:
                mask[:] = True
            cloud_masked = cloud[mask]
            color_masked = color_tmp[mask]

            # sample points
            if len(cloud_masked) >= self.num_point:
                idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]

            # convert data
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            end_points = dict()
            cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            cloud_sampled = cloud_sampled.to(self.device)
            end_points['point_clouds'] = cloud_sampled
            end_points['cloud_colors'] = color_sampled
            return end_points, cloud
        else:
            depth_tmp = depth.copy()
            cloud = create_point_cloud_from_depth_image(depth_tmp, self.cam_info[cam_name]['camera'], organized=True)
            mask = (self.cam_info[cam_name]['workspace_mask'] & (depth_tmp > 0))
            if not use_ws_mask:
                mask[:] = True
            cloud_masked = cloud[mask]

            # sample points
            if len(cloud_masked) >= self.num_point:
                idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]

            # convert data
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            end_points = dict()
            cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            cloud_sampled = cloud_sampled.to(self.device)
            end_points['point_clouds'] = cloud_sampled
            return end_points, cloud

    def get_grasps(self, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = self.model(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg, collision_mask

    def vis_grasps(self, gg, cloud, transform_into_world=True, cam_name='topview'):
        # visualise grasps in graspnet cam pose
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        if transform_into_world:
            old_cam_to_gg_mat = np.array([
                [*gg.rotation_matrices[0][0], gg.translations[0][0]],
                [*gg.rotation_matrices[0][1], gg.translations[0][1]],
                [*gg.rotation_matrices[0][2], gg.translations[0][2]],
                [0, 0, 0, 1]
            ])
            vis_cloud = deepcopy(cloud).transform(self.mujoco_cam_xy_plane_mirror_correction_mat)

            vis_gg = deepcopy(gg)
            vis_gg.translations[:, 2] *= -1
            rot_reflection_about_xy_plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rot_reflection_about_xy_plane_inv = np.linalg.inv(rot_reflection_about_xy_plane)
            vis_gg.rotation_matrices = rot_reflection_about_xy_plane_inv @ vis_gg.rotation_matrices

            y_rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            z_rot = Rotation.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()
            rot = y_rot @ z_rot
            rot_mat = np.array(
                [[*rot[0], 0],
                 [*rot[1], 0],
                 [*rot[2], 0],
                 [0, 0, 0, 1]]
            )
            eef_to_gg_rot_mat = Rotation.from_quat([0, 0, -0.707105, 0.707105]).as_matrix()
            eef_to_gg_mat = np.array(
                [[*eef_to_gg_rot_mat[0], 0.0],
                 [*eef_to_gg_rot_mat[1], 0.0],
                 [*eef_to_gg_rot_mat[2], 0.109],
                 [0, 0, 0, 1]]
            )
            gg_to_eef_mat = np.linalg.inv(eef_to_gg_mat)

            cam_to_gg_mat = self.mujoco_cam_xy_plane_mirror_correction_mat @ old_cam_to_gg_mat @ self.mujoco_cam_xy_plane_mirror_correction_mat @ rot_mat
            cam_to_eef_mat = cam_to_gg_mat @ gg_to_eef_mat

            world_cloud = deepcopy(vis_cloud).transform(self.cam_info[cam_name]['extrinsic'])
            world_to_gg_mat = self.cam_info[cam_name]['extrinsic'] @ cam_to_gg_mat
            world_to_eef_mat = self.cam_info[cam_name]['extrinsic'] @ cam_to_eef_mat

            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(
                self.cam_info[cam_name]['extrinsic'])
            world_to_gg_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(world_to_gg_mat)
            world_to_eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(world_to_eef_mat)
            o3d.visualization.draw_geometries(
                [world_cloud, world_to_gg_frame, world_to_eef_frame, world_frame, cam_frame])
