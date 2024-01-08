import torch
from torch.utils.data import Dataset
from data_helpers import dynamixel_data_preparation, vicon_data_preparation
import numpy as np


class ForwardKinematicsDataset(Dataset):
    """
    Dataset to learn the forward kinematics.
    """
    def __init__(self, file_name, norm=True, train=True):
        # Train and validation split
        self.train = train

        # Get the joint angle and end-effector position data from .csv file
        self.joint_angles = torch.from_numpy(dynamixel_data_preparation(file_name)).to(torch.float32)
        self.ee_rpy = vicon_data_preparation(file_name)
        self.ee = torch.from_numpy(self.ee_rpy).to(torch.float32)

        # Split data into training and validation set (90% to 10%)
        self.joint_angles_train = np.delete(self.joint_angles, np.arange(0, self.joint_angles.shape[0], 10), axis=0)
        self.ee_train = np.delete(self.ee, np.arange(0, self.ee.shape[0], 10), axis=0)
        self.joint_angles_val = self.joint_angles[0:self.joint_angles.shape[0]:10]
        self.ee_val = self.ee[0:self.ee.shape[0]:10]

        # Split data into training and validation data
        if self.train:
            self.joint_angles = self.joint_angles_train
            self.ee = self.ee_train
        else:
            self.joint_angles = self.joint_angles_val
            self.ee = self.ee_val

        if norm:
            # Define min and max joint angles
            self.joint_angles_min = 60  # samples between 60 and 120 degrees but limits 45 to 135
            self.joint_angles_max = 120

            self.joint_angles = (self.joint_angles - self.joint_angles_min) / (self.joint_angles_max - self.joint_angles_min)

        # Get number of samples in dataset
        self.n_samples = self.joint_angles.shape[0]

    def __getitem__(self, idx):
        x = self.joint_angles[idx, :]
        y = self.ee[idx, :]
        return x, y

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    fk_dataset_train = ForwardKinematicsDataset("data/231020_experiment_4_sync.csv", norm=False, train=True)
    fk_dataset_val = ForwardKinematicsDataset("data/231020_experiment_4_sync.csv", norm=False, train=False)

    print(len(fk_dataset_train))
    print(len(fk_dataset_val))

    for i in range(len(fk_dataset_train)):
        print(fk_dataset_train[i])
        print(fk_dataset_val[i])
