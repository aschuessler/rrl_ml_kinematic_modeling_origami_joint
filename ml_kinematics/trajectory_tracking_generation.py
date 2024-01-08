from inverse_kinematics_cross_entropy_method import cem_based_optimization
from forward_kinematics_nn import ForwardKinematicsNN
import torch
import pandas as pd
import numpy as np
import time


class Experiments:
    """
    Class for generating trajectories and calculating joint angles with cross entropy method
    """
    def __init__(self, file_name):
        self.file_name = file_name

        # Define forward kinematics neural network
        self.fk_nn_model_name = "models/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(self.fk_nn_model_name))
        self.fk_nn.eval()

        # Cross entropy method (CEM)
        self.n_samples = 1000
        self.num_opti_iters = 10
        self.num_elites = 100

        # Trajectory details
        self.samples = 20

    def calculation(self, trajectory, circle=False):
        # --- Calculate joint angles for roll trajectory ---
        if circle:
            traj_joint_angles = np.zeros((self.samples+int(self.samples/2), 4))
            traj_ypr_angles = np.zeros((self.samples+int(self.samples/2), 3))
            traj_loss = np.zeros(self.samples+int(self.samples/2))
            calculation_time = np.zeros(self.samples+int(self.samples/2))
        else:
            traj_joint_angles = np.zeros((self.samples, 4))
            traj_ypr_angles = np.zeros((self.samples, 3))
            traj_loss = np.zeros(self.samples)
            calculation_time = np.zeros(self.samples)

        for idx, target_angles in enumerate(trajectory):
            # Results with cross entropy method for inverse kinematics
            t_start_cal = time.time()
            joint_angles, loss, ypr_prediction = cem_based_optimization(self.fk_nn, target_angles, self.n_samples, self.num_opti_iters, self.num_elites)
            t_cal = time.time() - t_start_cal
            traj_joint_angles[idx, :] = joint_angles
            traj_ypr_angles[idx, :] = ypr_prediction
            traj_loss[idx] = loss
            calculation_time[idx] = t_cal

        return traj_joint_angles, traj_ypr_angles, traj_loss, calculation_time

    def save_experiment_to_csv(self, traj_type, traj, joint_angles, ypr_angles, loss, time):
        # Save trajectory data
        trajectory_data = {"target_yaw_angle": pd.Series(traj[:, 0]),
                           "predicted_yaw_angle": pd.Series(ypr_angles[:, 0]),
                           "target_pitch_angle": pd.Series(traj[:, 1]),
                           "predicted_pitch_angle": pd.Series(ypr_angles[:, 1]),
                           "target_roll_angle": pd.Series(traj[:, 2]),
                           "predicted_roll_angle": pd.Series(ypr_angles[:, 2]),
                           "joint_angle_1": pd.Series(joint_angles[:, 0]),
                           "joint_angle_2": pd.Series(joint_angles[:, 1]),
                           "joint_angle_3": pd.Series(joint_angles[:, 2]),
                           "joint_angle_4": pd.Series(joint_angles[:, 3]),
                           "loss": pd.Series(loss),
                           "time": pd.Series(time),
                           "mean_time": pd.Series(np.mean(time))}

        # Save to csv
        df_traj_roll = pd.DataFrame(trajectory_data)
        df_traj_roll.to_csv(f'experiments/{self.file_name}_{traj_type}.csv')

    def start(self):
        # Generate desired trajectories (yaw, pitch, roll)
        angle_range_roll_pitch = 40  # degrees
        angle_range_yaw = 15
        angle_range_circle = 40

        traj_steps_roll_pitch = angle_range_roll_pitch * torch.sin(torch.linspace(0.0, 2 * np.pi, self.samples)).reshape((self.samples, 1))
        traj_steps_yaw = angle_range_yaw * torch.sin(torch.linspace(0.0, 2 * np.pi, self.samples)).reshape((self.samples, 1))
        zeros = torch.zeros((self.samples, 1))
        traj_roll = torch.cat((zeros, zeros, traj_steps_roll_pitch), dim=1)
        traj_pitch = torch.cat((zeros, traj_steps_roll_pitch, zeros), dim=1)
        traj_yaw = torch.cat((traj_steps_yaw, zeros, zeros), dim=1)

        # Generate circular trajectory in roll-pitch plane (yaw = 0°)
        zeros_circle = torch.zeros((int(self.samples+self.samples/2), 1))
        traj_steps_circle_roll = torch.cat((torch.zeros((int(self.samples/4), 1)), angle_range_circle * torch.sin(torch.linspace(0.0, 2 * np.pi, self.samples)).reshape((self.samples, 1)), torch.zeros((int(self.samples/4), 1))), dim=0)
        traj_steps_circle_pitch = angle_range_circle * torch.sin(torch.linspace(0.0, 3 * np.pi, (self.samples+int(self.samples/2)))).reshape((self.samples+int(self.samples/2), 1))
        traj_circle = torch.cat((zeros_circle, traj_steps_circle_pitch, traj_steps_circle_roll), dim=1)

        # --- Calculate joint angles for roll trajectory ---
        traj_roll_joint_angles, traj_roll_ypr_angles, traj_roll_loss, traj_roll_time = self.calculation(traj_roll)
        self.save_experiment_to_csv("roll", traj_roll, traj_roll_joint_angles, traj_roll_ypr_angles, traj_roll_loss, traj_roll_time)
        print("Roll trajectory done!")

        # --- Calculate joint angles for pitch trajectory ---
        traj_pitch_joint_angles, traj_pitch_ypr_angles, traj_pitch_loss, traj_pitch_time = self.calculation(traj_pitch)
        self.save_experiment_to_csv("pitch", traj_pitch, traj_pitch_joint_angles, traj_pitch_ypr_angles, traj_pitch_loss, traj_pitch_time)
        print("Pitch trajectory done!")

        # --- Calculate joint angles for yaw trajectory ---
        traj_yaw_joint_angles, traj_yaw_ypr_angles, traj_yaw_loss, traj_yaw_time = self.calculation(traj_yaw)
        self.save_experiment_to_csv("yaw", traj_yaw, traj_yaw_joint_angles, traj_yaw_ypr_angles, traj_yaw_loss, traj_yaw_time)
        print("Yaw trajectory done!")

        # --- Calculate joint angles for circle trajectory ---
        traj_circle_joint_angles, traj_circle_ypr_angles, traj_circle_loss, traj_circle_time = self.calculation(traj_circle, circle=True)
        self.save_experiment_to_csv("circle", traj_circle, traj_circle_joint_angles, traj_circle_ypr_angles, traj_circle_loss, traj_circle_time)
        print("Circle trajectory done!")


if __name__ == "__main__":
    exp = Experiments("data_file_name")
    exp.start()
