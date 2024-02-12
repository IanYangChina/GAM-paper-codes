from manipulation_project.env.entangled_env import EntangledEnv
from manipulation_project.env.utils import get_real_depth_map
from manipulation_project.bpbot_agent.bpbot_agent import BPBot


# todo: tune bpbot parameters
def main():
    bpbot = BPBot(image_size=480, depth_max_distance=300, depth_min_distance=170, depth_rescale=1000)
    env = EntangledEnv(model_path='kuka_scene.xml', image_obs=True, depth=True, debug=True, num_hooks=3, image_obs_size=480)
    env._reset_sim(return_real_depth=False)
    t = 0
    while True:
        env._reset_sim(return_real_depth=False)
        rgb_1, depth_1 = env.render(mode='rgb_array', cam='straighttopview', width=env.image_obs_size, height=env.image_obs_size)
        depth_1 = get_real_depth_map(env.sim, depth_1)
        bpbot.detect_and_draw(depth_1)
        t += 1
        if t == 1000:
            break
    env.close()


if __name__ == '__main__':
    main()
