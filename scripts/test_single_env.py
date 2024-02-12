import os, time, json
import matplotlib as mlp
import matplotlib.pyplot as plt
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.env.wrappers import SingleEnvWrapper


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['num_hooks'] = 3
    env_config['hook_shape'] = 'C+'
    env_config['image_obs'] = False
    env_config['depth'] = False
    env_config['image_obs_size'] = 480
    env_config['use_graspnet'] = True
    env_config['use_collected_states'] = False
    env_config['sg_data_start_dir_ind'] = 0
    env_config['sg_data_end_dir_ind'] = 0
    env_config['sg_data_end_file_ind'] = 0
    env_config['sg_data_path'] = os.path.join(path, '..', 'SG_data')
    env_config['primitive'] = 'SixDofMovements'
    env_config['episode_length'] = 3
    env = SingleEnvWrapper(EntangledEnv,
                           config=env_config)
    # while True:
    #     obs = env.reset(repeat_last_state=False)
    #     env.env.render()
    done = False
    t = 0
    a = 0
    # mlp.use("TkAgg")
    # f, axarr = plt.subplots(2, 2)
    start = time.perf_counter()
    while True:
        # a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        env.env.render()
        a += 1
        if a == 12:
            a = 0
        if t % 3 == 0:
            obs = env.reset(repeat_last_state=False)
            env.env.render()
        # axarr[0][0].imshow(obs[:, :, :3])
        # axarr[0][1].imshow(info['depth'][:, :, 0])
        # axarr[1][0].imshow(obs[:, :, 3:])
        # axarr[1][1].imshow(info['depth'][:, :, 1])
        # plt.pause(0.00001)
        # if done:
        #     print(env.env.grasped_hook_id,
        #           info['finger_width'],
        #           info['current_timesteps'],
        #           info['is_object_dropped'],
        #           info['is_separated'],
        #           info['is_exceeded_workspace'])
            # f, axarr = plt.subplots(2, 2)
            # axarr[0][0].imshow(obs[:, :, :3])
            # axarr[0][1].imshow(info['depth'][:, :, 0])
            # axarr[1][0].imshow(obs[:, :, 3:])
            # axarr[1][1].imshow(info['depth'][:, :, 1])
            # plt.show()
            # plt.close()
            # plt.close()
            # f, axarr = plt.subplots(2, 2)

        t += 1
        if t == 5000:
            break

    print(time.perf_counter() - start)
    env.close()


if __name__ == '__main__':
    main()
