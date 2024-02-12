import os, json, time
import argparse
import numpy as np
from manipulation_project.rl_agent.dqn import DQN
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.env.wrappers import SingleEnvWrapper


def collect_labels(agent, eval_env, current_dir, d_i, f_i):
    separation_labels = np.zeros(shape=(10000,))
    object_dropped_labels = np.zeros(shape=(10000,))
    scene_alteration_values = np.zeros(shape=(10000,))
    num_timesteps = np.zeros(shape=(10000,))
    last_time = time.perf_counter()
    for dp_i in range(eval_env.data_loader.num_dp_per_file):
        eval_env.num_dones *= 0
        eval_env.num_ep_timesteps = 0
        eval_env.scene_alteration = 0
        eval_env.num_ep_timesteps = 0
        agent._interact(render=False, test=True, sleep=0)
        separation_labels[dp_i] = eval_env.num_dones[1]
        object_dropped_labels[dp_i] = eval_env.num_dones[2]
        scene_alteration_values[dp_i] = eval_env.scene_alteration
        num_timesteps[dp_i] = eval_env.num_ep_timesteps
        if (dp_i+1) % 1000 == 0:
            print("Passed 1000 Episodes, elapsed time: {:.2f} minutes".format((time.perf_counter()-last_time)/60))
            last_time = time.perf_counter()
    np.save(file=current_dir+'/good_grasps_separation_labels'+str(f_i)+'.npy',
            arr=np.array(separation_labels).astype(np.float))
    np.save(file=current_dir+'/good_grasps_object_dropped_labels'+str(f_i)+'.npy',
            arr=np.array(object_dropped_labels).astype(np.float))
    np.save(file=current_dir+'/good_grasps_scene_alteration_values'+str(f_i)+'.npy',
            arr=np.array(scene_alteration_values).astype(np.float))
    np.save(file=current_dir+'/good_grasps_num_timesteps'+str(f_i)+'.npy',
            arr=np.array(num_timesteps).astype(np.float))
    print("Saved the %i file for the %i dir" % (f_i+1, d_i+1))


def main(arguments):
    seed = arguments['seed']
    script_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_path, '..', 'results_'+arguments['hook_shape']+'_3', str(arguments['num_hooks'])+'hooks',
                        arguments['path'], 'seed'+str(seed))
    with open(os.path.join(path, 'configs', 'agent_config.json'), 'rb') as f_ac:
        agent_config = json.load(f_ac)
    agent_config['cuda_device_id'] = arguments['device_id']
    with open(os.path.join(path, 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['use_graspnet'] = False
    env_config['use_collected_states'] = True
    env_config['image_obs'] = False
    env_config['sg_data_start_dir_ind'] = 0
    env_config['sg_data_end_dir_ind'] = 4
    env_config['sg_data_end_file_ind'] = 4
    env_config['hook_shape'] = arguments['hook_shape']

    env_config['sg_data_path'] = os.path.join(script_path, '..', 'SG_data')
    eval_env = SingleEnvWrapper(EntangledEnv, config=env_config)
    agent = DQN(algo_params=agent_config, train_env=None, eval_env=eval_env, create_logger=False,
                path=path, seed=seed)
    agent._load_network(step=arguments['ckpt_step'])

    print("Start labelling...")
    eval_env.data_loader.start_dir_ind = 4
    eval_env.data_loader.start_file_ind = 0
    eval_env.data_loader.init()
    for d_i in [4]:
        current_dir = eval_env.data_loader.current_dir + ''
        for f_i in [0, 1, 2, 3, 4]:
            collect_labels(agent, eval_env, current_dir, d_i, f_i)

    eval_env.close()
    del eval_env, agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--nh', dest='num_hooks', type=int, default=3)
    parser.add_argument('--hs', dest='hook_shape', type=str, default='S')
    parser.add_argument('--path', dest='path', type=str, default='cs_coef_0.0_-1.0')
    parser.add_argument('--ckpt-step', dest='ckpt_step', type=int, default=int(2e6))
    parser.add_argument('--cuda', dest='device_id', default=0, type=int)
    args = vars(parser.parse_args())
    main(args)
