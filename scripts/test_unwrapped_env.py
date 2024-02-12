import os, time, json
import matplotlib as mlp
import matplotlib.pyplot as plt
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.graspnet_agent.grasp_estimator import GraspEstimator


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        config = json.load(f_ec)
    config['num_hooks'] = 3
    config['image_obs'] = False
    config['image_obs_size'] = 1200
    config['hook_shape'] = 'C+'
    env = EntangledEnv(config=config)
    # while True:
    #     _, _ = env._reset_sim(return_real_depth=False, return_color=False)
    #     env.render()

    grasp_estimator = GraspEstimator(cam_intrinsic=env.cam_intrinsic,
                                     cam_extrinsic=env.cam_extrinsic,
                                     factor_depth=1.0,
                                     img_height=env.render_height,
                                     img_width=env.render_width)
    # env._set_eef_start_pose(None)
    # obs = env.reset()
    done = True
    t = 0
    mlp.use("TkAgg")
    # f, axarr = plt.subplots(2, 2)
    start = time.perf_counter()
    while True:
        _, real_depth, c_1, c_2 = env._reset_sim(return_real_depth=True, return_color=True)
        rgb_1, depth_1 = env.render(mode='rgb_array', cam='backview',
                                    width=env.image_obs_size,
                                    height=env.image_obs_size)
        plt.imshow(rgb_1[::-1, ::-1, :])
        plt.xlabel(None)
        plt.xticks([])
        plt.ylabel(None)
        plt.yticks([])
        # plt.savefig(os.path.join(path, "../src/hooks.pdf"), bbox_inches='tight', dpi=500)
        enough_grasps_1, pregrasp_mjc_poses, gg_mjc_poses, _, _, _ = grasp_estimator.compute_grasp_pose_for_mjc_data(
            real_depth=real_depth[0],
            color=c_1,
            graspnet_to_mjc_gg_mat=env.graspnet_to_mjc_gg_mat,
            mjc_gg_to_pregrasp_mat=env.gg_to_pregrasp_mat,
            cam_name=env.graspnet_render_cams[0],
            num_grasps=10,
            visualisation=True)

        # a = env.action_space.sample()
        # obs, reward, done, info = env.step(a)
        # if done:
        #     env._reset_sim(return_real_depth=False)
            # for _ in range(5):
            #     env._set_eef_start_pose(None)
            # env.reset()
        # env.render()
        # axarr[0][0].imshow(obs[:, :, :3])
        # axarr[0][1].imshow(info['depth'][:, :, 0])
        # axarr[1][0].imshow(obs[:, :, 3:])
        # axarr[1][1].imshow(info['depth'][:, :, 1])
        # rgb_1, depth_1 = env.render(mode='rgb_array', cam='frontview',
        #                             width=env.image_obs_size,
        #                             height=env.image_obs_size)
        # plt.imshow(rgb_1[::-1, :, :])
        # plt.show()
        # plt.pause(0.00001)
        t += 1
        if t == 1000:
            break

    print(time.perf_counter() - start)
    env.close()


if __name__ == '__main__':
    main()
