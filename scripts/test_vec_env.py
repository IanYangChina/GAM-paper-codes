import os, time, json
from manipulation_project.env.wrappers import MpVecEnvWrapper
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['num_hooks'] = 4
    env_config['image_obs'] = True
    env_config['depth'] = False
    env_config['image_obs_size'] = 480
    env_config['use_graspnet'] = False
    env_config['use_collected_states'] = True
    env_config['sg_data_start_dir_ind'] = 0
    env_config['sg_data_end_dir_ind'] = 0
    env_config['sg_data_end_file_ind'] = 0
    env = MpVecEnvWrapper(sg_data_path=os.path.join(path, '..', 'SG_data'),
                          config=env_config, num_envs=10)
    obs = env.reset()
    t = 0
    start = time.perf_counter()
    # 1000 steps (seconds) | use-grasp-net | use-collected-states
    # state, 1 env, 172.65/184.48 | 97.60
    # state, 2 env, 152.43/148.75 | 63.66
    # state, 4 env, 116.64/128.50
    # state, 10 env, 143.19 | 19.54
    # rgb, 1 env, 226.00
    # rgb, 2 env, 169.64
    # rgb, 4 env, 163.33/138.10
    # rgb, 10 env, 142.42 | 39.64
    while not env.data_loader.finished:
        # env.render()
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        # env.render()
        # print(a)
        # print(obs.shape)
        # print(reward)
        # print(done)
        print(info)
        t += 1
        if t == 100:
            break

    print(time.perf_counter() - start)
    print(env.data_loader.num_passed_data_point)
    env.close()


if __name__ == '__main__':
    main()
