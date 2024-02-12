import os, json, time
import argparse
import logging
from manipulation_project.rl_agent.dqn import DQN
from manipulation_project.grasp_filter.grasp_filter import GraspFilter
from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.env.wrappers import SingleEnvWrapper, MpVecEnvWrapper
import numpy as np


def evaluate(script_path, arguments, g_filter, logger):
    result_dir = 'results_'
    if arguments['use_hemisphere_actions']:
        result_dir += 'hm_'
    if arguments['hook_shape'] != 'C':
        result_dir += arguments['hook_shape']
        result_dir += '_'
    result_dir += '3'

    path = os.path.join(script_path, '..', result_dir, str(arguments['num_hooks']) + 'hooks',
                        arguments['path'], 'seed11')
    with open(os.path.join(path, 'configs', 'env_config.json'), 'rb') as f_ec:
        env_config = json.load(f_ec)
    env_config.update({
        "image_obs": False,
        "use_graspnet": False,
        "hook_shape": arguments['hook_shape'],
        "use_collected_states": arguments['use_collected_states'],
        "primitive": 'SixDofMovements',
        "grip_ctrl": False,
        "end2end_rl": False,
    })
    if arguments['use_hemisphere_actions']:
        env_config['primitive'] = 'HemisphereMovements'
    env_config['episode_length'] = 3
    env_config['sg_data_path'] = os.path.join(script_path, '..', 'SG_data')
    eval_env = SingleEnvWrapper(EntangledEnv, config=env_config, grasp_filter=g_filter)
    eval_vec_env = MpVecEnvWrapper(sg_data_path=env_config['sg_data_path'], config=env_config,
                                   num_envs=arguments['num_env'],
                                   grasp_filter=g_filter)

    num_dones = []
    discarded_grasps = 0
    passed_grasps = 0
    scene_alteration = 0.0
    last_time = time.perf_counter()

    for seed in [11, 22, 33]:
        path = os.path.join(script_path, '..',
                            result_dir,
                            str(arguments['num_hooks']) + 'hooks',
                            arguments['path'],
                            'seed' + str(seed))

        with open(os.path.join(path, 'configs', 'agent_config.json'), 'rb') as f_ac:
            agent_config = json.load(f_ac)
        agent_config['cuda_device_id'] = arguments['device_id']
        agent_config['testing_episodes'] = arguments['num_episodes']

        ckpt_step = arguments['ckpt_step']
        agent = DQN(algo_params=agent_config, train_env=None, eval_env=eval_env, create_logger=False,
                    path=path, seed=seed)
        agent._load_network(step=ckpt_step)

        obs = eval_vec_env.reset()
        while eval_vec_env.num_passed_grasps <= arguments['num_episodes']:
            action = agent._select_action(obs, test=True)
            new_obs, reward, done, info = eval_vec_env.step(action)
            obs = new_obs

        passed_grasps += eval_vec_env.num_passed_grasps
        discarded_grasps += eval_vec_env.num_discarded_grasps
        num_dones.append(eval_vec_env.num_dones[1:] / eval_vec_env.num_dones[0])
        scene_alteration += (eval_vec_env.scene_alteration / eval_vec_env.num_dones[0])
        del agent

    eval_env.close()
    eval_vec_env.close()
    del eval_env, eval_vec_env

    logger.info("Finished, passed time: {:.2f} minutes".format((time.perf_counter() - last_time) / 60))
    logger.info("Percent of discarded grasps: {:.3f}".format(discarded_grasps/passed_grasps))
    logger.info("Average separation, object-dropped, exceeded-workspace & lift-failed over 3 seeds: {}".format(np.mean(num_dones, axis=0)))
    logger.info("Average scene alteration: {:.5f}".format(scene_alteration / 3))
    logger.info("------------------------------------")


def main(arguments):
    script_path = os.path.dirname(os.path.realpath(__file__))
    log_filename = os.path.dirname(__file__) + "/../results_gf/gf_evaluation_"+str(arguments['num_hooks'])+arguments['hook_shape']+"hooks.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if arguments['use_grasp_filter']:
        with open(os.path.join(script_path, '..', 'configs', 'grasp_filter_config.json'), 'rb') as f_ac:
            gf_config = json.load(f_ac)
        gf_config.update({
            "training": False,
            "num_hooks": arguments['num_hooks'],
            "hook_shape": arguments['hook_shape'],
        })
        g_filter = GraspFilter(path=os.path.join(script_path, '..',
                                                 'results_gf',
                                                 str(arguments['num_hooks']) + arguments['hook_shape'] + 'hooks_affordance_3ts'),
                               config=gf_config)
        g_filter.load_ckpt(epoch=10000)

        for con in [1]:
            g_filter.condition = con
            if con == 0:
                thres = [0.7, 0.8, 0.9, 0.95, 0.98]
            elif con == 1:
                thres = [0.3]
            elif con == 2:
                thres = [0.3]
            else:
                thres = [(0.8, 0.1), (0.8, 0.2), (0.9, 0.1), (0.9, 0.2)]

            for thre in thres:
                logger.info("New Run: using grasp filter, current condition {}, threshold {}".format(con, thre))
                if con == 0:
                    g_filter.threshold_1 = thre
                elif con == 1:
                    g_filter.threshold_2 = thre
                elif con == 2:
                    g_filter.threshold_3 = thre
                else:
                    g_filter.threshold_1 = thre[0]
                    g_filter.threshold_3 = thre[1]

                evaluate(script_path, arguments, g_filter=g_filter, logger=logger)
    else:
        if arguments['use_hemisphere_actions']:
            logger.info("New Run: no grasp filter, with hemisphere actions")
        else:
            logger.info("New Run: no grasp filter")
        evaluate(script_path, arguments, g_filter=None, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', type=str, default='cs_coef_0.0_-1.0')
    parser.add_argument('--render', dest='render', default=False, action='store_true')
    parser.add_argument('--num-ep', dest='num_episodes', type=int, default=15000)
    parser.add_argument('--ckpt-step', dest='ckpt_step', type=int, default=int(2e6))
    parser.add_argument('--cuda', dest='device_id', default=0, type=int)
    parser.add_argument('--cs', dest='use_collected_states', default=True, action='store_true')
    parser.add_argument('--gf', dest='use_grasp_filter', default=False, action='store_true')
    parser.add_argument('--hs', dest='hook_shape', default='S', type=str)
    parser.add_argument('--nenv', dest='num_env', default=5, type=int)
    parser.add_argument('--nh', dest='num_hooks', default=2, type=int)
    parser.add_argument('--hm', dest='use_hemisphere_actions', default=False, action='store_true')
    args = vars(parser.parse_args())
    main(args)
