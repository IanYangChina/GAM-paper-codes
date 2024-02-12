import os, json, argparse
from manipulation_project.grasp_filter.grasp_filter import GraspFilter


def main(args):
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, '..', 'configs', 'grasp_filter_config.json'), 'rb') as f_ac:
        gf_config = json.load(f_ac)

    gf_config.update({
        "training": True,
        "num_hooks": args['num_hooks'],
        "hook_shape": args['hook_shape'],
        "learning_rate": 0.001,
        "batch_size": 1024,
        "num_epoch": 10001,
        "log_interval": 10,
        "switch_file_interval": 1,
        "saving_interval": 1000,
    })
    g_filter = GraspFilter(path=os.path.join(path, '..',
                                             'results_gf',
                                             str(args['num_hooks'])+args['hook_shape']+'hooks_affordance_3ts'),
                           config=gf_config)
    g_filter.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nh', dest='num_hooks', default=3, type=int)
    parser.add_argument('--hs', dest='hook_shape', default='C+', type=str)
    args = vars(parser.parse_args())
    main(args)
