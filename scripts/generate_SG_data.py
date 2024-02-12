import os, time, json
import numpy as np
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.graspnet_agent.grasp_estimator import GraspEstimator
import pickle
from copy import deepcopy as dcp
import argparse


def main(arguments):
    data_file_path = os.path.join(os.path.dirname(__file__),
                                  '..',
                                  'SG_data',
                                  str(arguments['num-hook'])+'_hooks_'+arguments['hook-shape'],
                                  'dir_'+str(arguments['dir-id']))
    os.makedirs(data_file_path, exist_ok=True)
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['num_hooks'] = arguments['num-hook']
    env_config['hook_shape'] = arguments['hook-shape']
    env = EntangledEnv(config=env_config)

    grasp_estimator = GraspEstimator(cam_intrinsic=env.cam_intrinsic,
                                     cam_extrinsic=env.cam_extrinsic,
                                     factor_depth=1.0,
                                     img_height=env.render_height,
                                     img_width=env.render_width,
                                     cuda_device_id=0)
    num_saved_good_data_point = arguments['good-start-ind']
    num_saved_bad_data_point = arguments['bad-start-ind']

    num_datapoint = 0
    num_good_state = 0
    num_bad_state = 0
    start = time.perf_counter()
    good_grasps = []
    bad_grasps = []
    good_grasp_features = []
    bad_grasp_features = []
    good_grasp_mjc_states = []
    bad_grasp_mjc_states = []
    done = False
    while not done:
        pregrasp_mjc_poses, gg_mjc_poses = np.array([None]), np.array([None])
        enough_grasps = False
        while not enough_grasps:
            _, real_depth = env._reset_sim(return_real_depth=True)
            enough_grasps, pregrasp_mjc_poses, gg_mjc_poses, grasp_point_idx, grasp_pos, vp_features = grasp_estimator.compute_grasp_pose_for_mjc_data(
                real_depth=real_depth[0],
                graspnet_to_mjc_gg_mat=env.graspnet_to_mjc_gg_mat,
                mjc_gg_to_pregrasp_mat=env.gg_to_pregrasp_mat,
                cam_name=env.graspnet_render_cams[0],
                num_grasps=20,
                visualisation=False)

        for i in range(pregrasp_mjc_poses.shape[0]):
            success_grasp, mjc_sim_state = env._set_eef_start_pose(pregrasp_pose=pregrasp_mjc_poses[i],
                                                                   gg_pose=gg_mjc_poses[i])

            if success_grasp:
                good_grasps.append(grasp_pos[i])
                good_grasp_features.append(vp_features[:, grasp_point_idx[i]])
                good_grasp_mjc_states.append(dcp(mjc_sim_state))
                num_good_state += 1
                if num_good_state % 10000 == 0 and num_good_state != 0:
                    print("Passed %i datapoints, good states: %i, bad states %i" % (num_datapoint, num_good_state, num_bad_state))
                    print("Saving good grasp data...")
                    np.save(file=data_file_path+'/good_grasps_'+str(num_saved_good_data_point)+'.npy', arr=np.array(good_grasps).astype(np.float))
                    np.save(file=data_file_path+'/good_grasp_features_'+str(num_saved_good_data_point)+'.npy', arr=np.array(good_grasp_features).astype(np.float))
                    with open(data_file_path+'/good_grasp_mjc_states_'+str(num_saved_good_data_point)+'.mjc', 'wb') as f:
                        pickle.dump(good_grasp_mjc_states, f)
                    good_grasps = []
                    good_grasp_features = []
                    good_grasp_mjc_states = []
                    num_saved_good_data_point += 1
            else:
                if num_saved_bad_data_point < 5:
                    bad_grasps.append(grasp_pos[i])
                    bad_grasp_features.append(vp_features[:, grasp_point_idx[i]])
                    bad_grasp_mjc_states.append(dcp(mjc_sim_state))
                    num_bad_state += 1
                    if num_bad_state % 10000 == 0 and num_bad_state != 0:
                        print("Passed %i datapoints, good states: %i, bad states %i" % (num_datapoint, num_good_state, num_bad_state))
                        print("Saving bad grasp data...")
                        np.save(file=data_file_path+'/bad_grasps_'+str(num_saved_bad_data_point)+'.npy', arr=np.array(bad_grasps).astype(np.float))
                        np.save(file=data_file_path+'/bad_grasp_features_'+str(num_saved_bad_data_point)+'.npy', arr=np.array(bad_grasp_features).astype(np.float))
                        with open(data_file_path+'/bad_grasp_mjc_states_'+str(num_saved_bad_data_point)+'.mjc', 'wb') as f:
                            pickle.dump(bad_grasp_mjc_states, f)
                        bad_grasps = []
                        bad_grasp_features = []
                        bad_grasp_mjc_states = []
                        num_saved_bad_data_point += 1

            num_datapoint += 1
            if num_saved_good_data_point == 5:
                break
        if num_saved_good_data_point == 5:
            break

    print(time.perf_counter() - start)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-id', dest='dir-id', default=0, type=int)
    parser.add_argument('--num-hook', dest='num-hook', default=3, type=int)
    parser.add_argument('--gs-ind', dest='good-start-ind', default=0, type=int)
    parser.add_argument('--bs-ind', dest='bad-start-ind', default=0, type=int)
    parser.add_argument('--hs', dest='hook-shape', default='S', type=str)
    args = vars(parser.parse_args())
    main(args)
