"""This script test the straight-up lifting motion (0.15m) for TAGs"""

import os, json, time
import argparse
from manipulation_project.env.wrappers import MpVecEnvWrapper
import numpy as np


def main(arguments):
    script_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['use_graspnet'] = False
    env_config['use_collected_states'] = True
    env_config['primitive'] = 'EmptyMovement'
    env_config['episode_length'] = 1
    env_config['sg_data_path'] = os.path.join(script_path, '..', 'SG_data')

    for nh in [2, 3, 4]:
        print("Testing straight-up motion for " + str(nh) + "-C-hook task...")
        last_time = time.perf_counter()
        num_dones = []
        scene_alteration = 0.0
        env_config['num_hooks'] = nh
        env_config['hook_shape'] = 'C'
        # eval_env = SingleEnvWrapper(EntangledEnv, config=env_config)
        eval_env = MpVecEnvWrapper(sg_data_path=env_config['sg_data_path'], config=env_config, num_envs=arguments['num_env'])
        eval_env.reset()
        while eval_env.num_dones[0] <= arguments['num_episodes']:
            eval_env.step([0 for _ in range(arguments['num_env'])])

        num_dones.append(eval_env.num_dones[1:] / arguments['num_episodes'])
        scene_alteration += (eval_env.scene_alteration / arguments['num_episodes'])
        eval_env.close()
        del eval_env

        print("Finished, passed time: %0.2f minutes" % ((time.perf_counter() - last_time) / 60))
        print("Average separation, object-dropped, exceeded-workspace & lift-failed over " + str(arguments['num_episodes']) + " episodes: ", np.mean(num_dones, axis=0))
        print("Average scene alteration: ", scene_alteration)

    for hs in ['C+', 'S']:
        print("Testing straight-up motion for 3-" + hs + "-hook task...")
        last_time = time.perf_counter()
        num_dones = []
        scene_alteration = 0.0
        env_config['num_hooks'] = 3
        env_config['hook_shape'] = hs
        eval_env = MpVecEnvWrapper(sg_data_path=env_config['sg_data_path'], config=env_config, num_envs=arguments['num_env'])
        eval_env.reset()
        while eval_env.num_dones[0] <= arguments['num_episodes']:
            eval_env.step([0 for _ in range(arguments['num_env'])])

        num_dones.append(eval_env.num_dones[1:] / arguments['num_episodes'])
        scene_alteration += (eval_env.scene_alteration / arguments['num_episodes'])
        eval_env.close()
        del eval_env

        print("Finished, passed time: %0.2f minutes" % ((time.perf_counter() - last_time) / 60))
        print("Average separation, object-dropped, exceeded-workspace & lift-failed over " + str(arguments['num_episodes']) + " episodes: ", np.mean(num_dones, axis=0))
        print("Average scene alteration: ", scene_alteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ep', dest='num_episodes', type=int, default=50000)
    parser.add_argument('--nenv', dest='num_env', default=5, type=int)
    args = vars(parser.parse_args())
    main(args)
