import torch as T
import torch.nn as nn
import torch.nn.functional as F
from drl_implementation.agent.utils.networks_mlp import orthogonal_init


class AffordanceNet(nn.Module):
    def __init__(self, grasp_pose_dim, grasp_feature_dim, fc1_size=512, fc2_size=512, fc3_size=256):
        super(AffordanceNet, self).__init__()
        self.bn = nn.BatchNorm1d(grasp_pose_dim+grasp_feature_dim)
        self.fc1 = nn.Linear(grasp_pose_dim+grasp_feature_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.out = nn.Linear(fc3_size, 3)
        self.apply(orthogonal_init)

    def forward(self, inputs):
        x = F.relu(self.fc1(self.bn(inputs)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.out(x)
        is_separated = T.sigmoid(out[:, 0])
        is_dropped = T.sigmoid(out[:, 1])
        scene_alteration_scores = out[:, 2]
        return is_separated, is_dropped, scene_alteration_scores


class ReachNet(nn.Module):
    def __init__(self, grasp_pose_dim, grasp_feature_dim, fc1_size=512, fc2_size=512, fc3_size=256):
        super(ReachNet, self).__init__()
        self.bn = nn.BatchNorm1d(grasp_pose_dim+grasp_feature_dim)
        self.fc1 = nn.Linear(grasp_pose_dim+grasp_feature_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.out = nn.Linear(fc3_size, 1)
        self.apply(orthogonal_init)

    def forward(self, inputs):
        x = F.relu(self.fc1(self.bn(inputs)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.out(x)
        is_reachable = T.sigmoid(out)
        return is_reachable
