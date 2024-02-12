import numpy as np
from scipy.spatial.transform import Rotation

rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
print(np.concatenate([[rot[-1]], rot[:-1]]))

rot_ = Rotation.from_quat([0.06162842, 0.06162842, 0.70441603, 0.70441603]).as_euler('xyz', degrees=True)
print(rot_)

# import torch as T
# device = T.device("cuda")
# values = T.tensor([[2, 3, 5, 6, 8],
#                    [5, 4, 11, 8, 9]], device=device)
# a = T.argmax(values, dim=1).tolist()
# print(a)
# exit()
