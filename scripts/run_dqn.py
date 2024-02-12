import os
import json
import argparse
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.env.wrappers import SingleEnvWrapper, MpVecEnvWrapper
from manipulation_project.rl_agent.dqn import DQN


def main(arguments):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'agent_config.json'), 'rb') as f_ac:
        agent_config = json.load(f_ac)
    agent_config['critic_learning_rate'] = arguments['learning_rate']
    agent_config['cuda_device_id'] = arguments['device_id']

    with open(os.path.join(path, '..', 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config['end2end_rl'] = arguments['end2end_rl']
    env_config['use_graspnet'] = arguments['use_graspnet']
    env_config['use_collected_states'] = arguments['use_collected_states']
    env_config['image_obs'] = arguments['image_obs']
    env_config['num_hooks'] = arguments['num_hooks']
    env_config['hook_shape'] = arguments['hook_shape']
    env_config['coef_r_distance'] = arguments['coef-rd']
    env_config['coef_r_scene_alteration'] = arguments['coef-rsa']
    
    env_config['episode_length'] = arguments['episode_length']
    result_folder = 'results_' + env_config['hook_shape'] + '_' + str(env_config['episode_length'])
    if arguments['use_hemisphere_actions']:
        env_config['primitive'] = 'HemisphereMovements'
        result_folder = 'results_hm_' + env_config['hook_shape'] + '_' + str(env_config['episode_length'])

    suffix = ''
    if arguments['use_graspnet']:
        assert not arguments['use_collected_states']
        suffix = 'gn_' + suffix
    if arguments['use_collected_states']:
        assert not arguments['use_graspnet']
        suffix = 'cs_' + suffix
    if arguments['image_obs']:
        suffix = 'img_' + suffix
    suffix = suffix + 'coef_' + str(arguments['coef-rd']) + '_' + str(arguments['coef-rsa'])
    dqn_path = os.path.join(path, '..', result_folder, str(env_config['num_hooks'])+'hooks', suffix)
    sg_data_path = os.path.join(path, '..', 'SG_data')
    env_config['sg_data_path'] = os.path.join(path, '..', 'SG_data')

    seeds = [11, 22, 33]
    for seed in seeds:
        env_config['seed'] = seed

        eval_env = SingleEnvWrapper(EntangledEnv, config=env_config)
        train_env = MpVecEnvWrapper(sg_data_path=sg_data_path, config=env_config, num_envs=arguments['num_env'])

        seed_path = dqn_path + '/seed'+str(seed)
        os.makedirs(seed_path+'/configs', exist_ok=True)
        with open(os.path.join(seed_path, 'configs', 'agent_config.json'), 'w') as f_ac:
            json.dump(agent_config, f_ac)
        with open(os.path.join(seed_path, 'configs', 'env_config.json'), 'w') as f_ac:
            json.dump(env_config, f_ac)

        agent = DQN(algo_params=agent_config, train_env=train_env, eval_env=eval_env,
                    path=seed_path, seed=seed)
        agent.run(test=False)

        train_env.close()
        eval_env.close()
        del train_env, eval_env, agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest='image_obs', default=False, action='store_true')
    parser.add_argument('--nenv', dest='num_env', default=5, type=int)
    parser.add_argument('--e2erl', dest='end2end_rl', default=False, action='store_true')
    parser.add_argument('--gn', dest='use_graspnet', default=False, action='store_true')
    parser.add_argument('--cs', dest='use_collected_states', default=True, action='store_true')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('--cuda', dest='device_id', default=0, type=int)
    parser.add_argument('--seed', dest='seed', default=11, type=int)
    parser.add_argument('--hs', dest='hook_shape', default='S', type=str)
    parser.add_argument('--nh', dest='num_hooks', default=3, type=int)
    parser.add_argument('--coef-rd', dest='coef-rd', default=0.0, type=float)
    parser.add_argument('--coef-rsa', dest='coef-rsa', default=-1.0, type=float)
    parser.add_argument('--hm', dest='use_hemisphere_actions', default=False, action='store_true')
    parser.add_argument('--ep-len', dest='episode_length', default=3, type=int)
    args = vars(parser.parse_args())
    main(args)
