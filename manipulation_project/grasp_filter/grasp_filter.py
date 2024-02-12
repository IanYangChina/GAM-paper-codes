import os
import numpy as np
import torch as T
import torch.nn.functional as F
from manipulation_project.grasp_filter.network import AffordanceNet
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from manipulation_project.env.sg_data_loader import DataLoader


class GraspFilter(object):
    def __init__(self, path, config):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.affordance_net = AffordanceNet(grasp_pose_dim=config['grasp_pose_dim'],
                                            grasp_feature_dim=config['grasp_feature_dim']).to(self.device)
        self.affordance_net_optimiser = Adam(self.affordance_net.parameters(), lr=config['learning_rate'])
        self.batch_size = config['batch_size']
        self.num_epoch = config['num_epoch']
        self.log_interval = config['log_interval']
        self.saving_interval = config['saving_interval']
        self.switch_file_interval = config['switch_file_interval']
        self.training = config['training']
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.ckpt_path = self.path + '/ckpt'
        os.makedirs(self.ckpt_path, exist_ok=True)
        if self.training:
            self.logger = SummaryWriter(log_dir=self.path)
        else:
            self.affordance_net.eval()
        self.training_data_loader = DataLoader(path=path + '/../../SG_data', num_hooks=config['num_hooks'],
                                               bad_grasps=False, mani_labels=True,
                                               start_dir_ind=0, end_dir_ind=4,
                                               start_file_ind=0, end_file_ind=4)
        self.testing_data_loader = DataLoader(path=path + '/../../SG_data', num_hooks=config['num_hooks'],
                                              bad_grasps=False, mani_labels=True,
                                              start_dir_ind=0, end_dir_ind=4,
                                              start_file_ind=0, end_file_ind=4)
        self.condition = config['condition']
        assert self.condition < 4, "only 3 conditions are supported"
        self.threshold_1 = config['threshold_1']  # for separation
        self.threshold_2 = config['threshold_2']  # for object dropped
        self.threshold_3 = config['threshold_3']  # for scene alteration

    def run(self):
        self.training_data_loader.init()
        self.testing_data_loader.init()
        for n in range(self.num_epoch):
            self.train_affordance_net(n)

    def train_affordance_net(self, n):
        grasps, features, object_dropped, separation, scene_alteration = \
            self.training_data_loader.get_good_random_batch(batch_size=self.batch_size)

        inputs = np.concatenate((grasps, features), axis=1)
        inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
        is_separated, is_dropped, scene_alteration_scores = self.affordance_net(inputs)
        separation = T.as_tensor(separation, dtype=T.float, device=self.device).view(self.batch_size, 1)
        object_dropped = T.as_tensor(object_dropped, dtype=T.float, device=self.device).view(self.batch_size, 1)
        scene_alteration = T.as_tensor(scene_alteration, dtype=T.float, device=self.device).view(self.batch_size, 1)
        self.affordance_net_optimiser.zero_grad()
        loss_1 = F.binary_cross_entropy(is_separated.view(self.batch_size, 1), separation)
        loss_2 = F.binary_cross_entropy(is_dropped.view(self.batch_size, 1), object_dropped)
        loss_3 = F.smooth_l1_loss(scene_alteration_scores.view(self.batch_size, 1), scene_alteration)
        (loss_1 + loss_2 + loss_3).backward()
        self.affordance_net_optimiser.step()

        if n != 0 and n % self.log_interval == 0:
            self.logger.add_scalar(tag='Loss/separation', scalar_value=loss_1, global_step=n)
            self.logger.add_scalar(tag='Loss/object_dropped', scalar_value=loss_2, global_step=n)
            self.logger.add_scalar(tag='Loss/scene_alteration', scalar_value=loss_3, global_step=n)
            self.test_affordance_net(n)
            self.testing_data_loader.load_next_file()
        if n != 0 and n % self.switch_file_interval == 0:
            self.training_data_loader.load_next_file()
        if n != 0 and n % self.saving_interval == 0:
            self.save_ckpt(n)

    def test_affordance_net(self, n):
        grasps, features, object_dropped, separation, scene_alteration = \
            self.testing_data_loader.get_good_random_batch(batch_size=self.batch_size)
        inputs = np.concatenate((grasps, features), axis=1)
        inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
        with T.no_grad():
            is_separated, is_dropped, scene_alteration_scores = self.affordance_net(inputs)
        separation = T.as_tensor(separation, dtype=T.float, device=self.device).view(self.batch_size, 1)
        object_dropped = T.as_tensor(object_dropped, dtype=T.float, device=self.device).view(self.batch_size, 1)
        scene_alteration = T.as_tensor(scene_alteration, dtype=T.float, device=self.device).view(self.batch_size, 1)
        loss_1 = F.binary_cross_entropy(is_separated.view(self.batch_size, 1), separation)
        loss_2 = F.binary_cross_entropy(is_dropped.view(self.batch_size, 1), object_dropped)
        loss_3 = F.smooth_l1_loss(scene_alteration_scores.view(self.batch_size, 1), scene_alteration)
        self.logger.add_scalar(tag='TestLoss/separation', scalar_value=loss_1, global_step=n)
        self.logger.add_scalar(tag='TestLoss/object_dropped', scalar_value=loss_2, global_step=n)
        self.logger.add_scalar(tag='TestLoss/scene_alteration', scalar_value=loss_3, global_step=n)

    def save_ckpt(self, epoch):
        T.save(self.affordance_net.state_dict(), self.ckpt_path + '/an_ckpt_' + str(epoch) + '.pt')

    def load_ckpt(self, epoch):
        self.affordance_net.load_state_dict(T.load(self.ckpt_path + '/an_ckpt_' + str(epoch) + '.pt',
                                                   map_location=self.device))

    def evaluate(self, grasps, features):
        inputs = np.concatenate((grasps, features), axis=1)
        inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
        with T.no_grad():
            is_separated, is_dropped, scene_alteration_scores = self.affordance_net(inputs)
        return is_separated, is_dropped, scene_alteration_scores
