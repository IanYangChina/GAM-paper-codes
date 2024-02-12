import numpy as np
from PIL import Image
import os
from manipulation_project.graspnet_agent.graspnet_checkpoints import checkpoint_dir
workspace_mask = np.array(Image.open(os.path.join(checkpoint_dir, 'topview_workspace_mask.png')))
new_mask = workspace_mask.copy()[:720, :720]
new_mask[:] = False
new_mask[96:618, 86:622] = True
new_mask = Image.fromarray(new_mask).save(os.path.join(checkpoint_dir, 'frontview_workspace_mask.png'))
exit()
