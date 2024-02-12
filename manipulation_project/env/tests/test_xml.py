import os
import numpy as np
from manipulation_project.env.assets import asset_dir
from mujoco_py import MjSim, MjViewer, load_model_from_path
from manipulation_project.env.utils import ctrl_set_action


fullpath = os.path.join(os.path.dirname(__file__), '..', 'assets', 'scene', 'kuka_scene_S.xml')
model = load_model_from_path(fullpath)
sim = MjSim(model, nsubsteps=20)
viewer = MjViewer(sim)
l_finger_tip_xpos = sim.data.get_site_xpos('l_finger_tip_site')
r_finger_tip_xpos = sim.data.get_site_xpos('r_finger_tip_site')
finger_width = np.linalg.norm(l_finger_tip_xpos - r_finger_tip_xpos, axis=-1)
print(finger_width)
fw_range = [-0.02, 0.2]
a = 0
fw_ctrl = 0.03
# sim.data.ctrl[:] = [fw_ctrl, -fw_ctrl]
sim.step()

for i in range(1000):
    sim.step()
    viewer.render()
