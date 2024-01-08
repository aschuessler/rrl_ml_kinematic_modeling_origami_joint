import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from forward_kinematics_nn import ForwardKinematicsNN


class FigureMaker:
    """
    Class for generating figures for the feedback response experiments
    """
    def __init__(self, file_name_actual=None, file_name_desired=None, file_name_sync=None):
        # Open csv file as pandas dataframe
        self.file_name_actual = file_name_actual
        self.file_name_desired = file_name_desired
        self.file_name_sync = file_name_sync
        self.df_actual = pd.read_csv(self.file_name_actual)
        self.df_desired = pd.read_csv(self.file_name_desired)
        self.df_sync = pd.read_csv(self.file_name_sync)

        # Define forward kinematics neural network
        self.fk_nn_model_name = "models/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(f'{self.fk_nn_model_name}'))
        self.fk_nn.eval()

    def file_synchronisation(self, file_name_save):

        # Get timestamp at start and end of the files
        timestamp_start_actual = self.df_actual["timestamp"][0]
        timestamp_start_desired = self.df_desired["timestamp"][0]
        timestamp_end_actual = self.df_actual["timestamp"].iloc[-1]
        timestamp_end_desired = self.df_desired["timestamp"].iloc[-1]

        # Take the timestamp of the file that was started at a later (higher) timestamp
        if timestamp_start_actual > timestamp_start_desired:
            timestamp_start_target = timestamp_start_actual
        else:
            timestamp_start_target = timestamp_start_desired

        # Take the timestamp of the file that was finshed at a earlier (lower) timestep
        if timestamp_end_actual > timestamp_end_desired:
            timestamp_end_target = timestamp_end_desired
        else:
            timestamp_end_target = timestamp_end_actual

        for index, row in self.df_actual.iterrows():
            if timestamp_end_target < row["timestamp"] < timestamp_start_target:
                self.df_actual = self.df_actual.drop(index)

        for index, row in self.df_desired.iterrows():
            if timestamp_end_target < row["timestamp"] < timestamp_start_target:
                self.df_desired.drop(index)

        self.df_actual = self.df_actual.reset_index()
        self.df_desired = self.df_desired.reset_index()

        # Add the matching vicon data to the dynamixel dataframe
        timestamp_desired = []
        roll_desired = []
        pitch_desired = []
        yaw_desired = []

        print(len(self.df_actual))
        print(len(self.df_desired))

        for actual_idx, actual_row in self.df_actual.iterrows():
            print(f"Dynamixel file row: {actual_idx}")

            time_diff = [np.abs(actual_row["timestamp"] - desired_row["timestamp"]) for desired_idx, desired_row in self.df_desired.iterrows()]

            desired_matching_idx = np.argmin(time_diff)
            timestamp_desired.append(self.df_desired["timestamp"][desired_matching_idx])

            roll_desired.append(self.df_desired["roll_des"][desired_matching_idx])
            pitch_desired.append(self.df_desired["pitch_des"][desired_matching_idx])
            yaw_desired.append(self.df_desired["yaw_des"][desired_matching_idx])

        self.df_actual["timestamp_desired"] = pd.Series(timestamp_desired)
        self.df_actual["roll_desired"] = pd.Series(roll_desired)
        self.df_actual["pitch_desired"] = pd.Series(pitch_desired)
        self.df_actual["yaw_desired"] = pd.Series(yaw_desired)

        # Save the synchronised dataframes to a .csv file
        self.df_actual.to_csv(file_name_save, index=False)

        print("File Synchronisation Finished!")

    def plot_feedback_response(self):
        # Create a figure with three subplots stacked vertically (roll, pitch and yaw angle)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

        # First timestamp
        t_0 = self.df_sync["timestamp"][1]
        joint_angles = np.array([self.df_sync["joint_angle_1"], self.df_sync["joint_angle_2"],
                                 self.df_sync["joint_angle_3"], self.df_sync["joint_angle_4"]]).swapaxes(0, 1)
        print(joint_angles.shape)
        # Normalize joint angles
        joint_angles_min = 60  # samples between 60 and 120 degrees but limits 45 to 135
        joint_angles_max = 120
        joint_angles_norm = (joint_angles - joint_angles_min) / (joint_angles_max - joint_angles_min)

        with torch.no_grad():
            ypr_prediction = self.fk_nn(torch.tensor(joint_angles_norm, dtype=torch.float32))

        print(ypr_prediction)

        # Roll trajectory
        ax1.plot(self.df_sync["timestamp"] - t_0, self.df_sync["roll_desired"], c="black", linestyle="dashed", label="Target")
        ax1.plot(self.df_sync["timestamp"] - t_0, ypr_prediction[:, 2], c="black", label="Predicted angle")
        ax1.set_xlim([0.0, 20.0])
        ax1.set_ylabel('Roll / °')
        ax1.set_ylim([0.0, 80.0])
        ax1.legend()

        # Pitch trajectory
        ax2.plot(self.df_sync["timestamp"] - t_0, self.df_sync["pitch_desired"], c="black", linestyle="dashed", label="Target")
        ax2.plot(self.df_sync["timestamp"] - t_0, ypr_prediction[:, 1], c="black", label="Predicted angle")
        ax2.set_xlim([0.0, 20.0])
        ax2.set_ylabel('Pitch / °')
        ax2.set_ylim([0.0, 80.0])
        ax2.legend()

        # Pitch trajectory
        ax3.plot(self.df_sync["timestamp"] - t_0, self.df_sync["yaw_desired"], c="black", linestyle="dashed", label="Target")
        ax3.plot(self.df_sync["timestamp"] - t_0, ypr_prediction[:, 0], c="black", label="Predicted angle")
        ax3.set_xlim([0.0, 20.0])
        ax3.set_ylabel('Yaw / °')
        ax3.set_ylim([0.0, 80.0])
        ax3.legend()

        #fig.savefig(save_figure_name, dpi=600)

    def plot_feedback_response_roll(self):
        # Create a figure with three subplots stacked vertically (roll, pitch and yaw angle)
        plt.figure(figsize=(8, 6))

        # First timestamp
        t_0 = self.df_sync["timestamp"][1]
        joint_angles = np.array([self.df_sync["joint_angle_1"], self.df_sync["joint_angle_2"],
                                 self.df_sync["joint_angle_3"], self.df_sync["joint_angle_4"]]).swapaxes(0, 1)

        # Normalize joint angles
        joint_angles_min = 60  # samples between 60 and 120 degrees but limits 45 to 135
        joint_angles_max = 120
        joint_angles_norm = (joint_angles - joint_angles_min) / (joint_angles_max - joint_angles_min)

        with torch.no_grad():
            ypr_prediction = self.fk_nn(torch.tensor(joint_angles_norm, dtype=torch.float32))

        # Roll trajectory
        plt.plot(self.df_sync["timestamp"] - t_0 - 7.0, self.df_sync["roll_desired"], c="black", linestyle="dashed", label="Target angle")
        plt.plot(self.df_sync["timestamp"] - t_0 - 7.0, ypr_prediction[:, 2], c="black", label="Predicted angle")
        plt.plot(self.df_sync["timestamp"] - t_0 - 7.0, np.ones_like(self.df_sync["timestamp"]) * 20.0 * 0.1, c="black")
        plt.plot(self.df_sync["timestamp"] - t_0 - 7.0, np.ones_like(self.df_sync["timestamp"]) * 20.0 * 0.9, c="black")
        plt.xlim([0.0, 5.0])
        ticks = np.arange(0.0, 5.5, 0.5)
        plt.xticks(ticks, fontsize=14)
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Roll Angle (°)', fontsize=18)
        plt.ylim([-5.0, 25.0])
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, loc="upper left")

    def plot_feedback_response_pitch(self):
        # Create a figure with three subplots stacked vertically (roll, pitch and yaw angle)
        plt.figure(figsize=(8, 6))

        # First timestamp
        t_0 = self.df_sync["timestamp"][1]
        joint_angles = np.array([self.df_sync["joint_angle_1"], self.df_sync["joint_angle_2"],
                                 self.df_sync["joint_angle_3"], self.df_sync["joint_angle_4"]]).swapaxes(0, 1)

        # Normalize joint angles
        joint_angles_min = 60  # samples between 60 and 120 degrees but limits 45 to 135
        joint_angles_max = 120
        joint_angles_norm = (joint_angles - joint_angles_min) / (joint_angles_max - joint_angles_min)

        with torch.no_grad():
            ypr_prediction = self.fk_nn(torch.tensor(joint_angles_norm, dtype=torch.float32))

        # Pitch trajectory
        plt.plot(self.df_sync["timestamp"] - t_0 - 8.0, self.df_sync["pitch_desired"], c="black", linestyle="dashed", label="Target angle")
        plt.plot(self.df_sync["timestamp"] - t_0 - 8.0, ypr_prediction[:, 1], c="black", label="Predicted angle")
        plt.plot(self.df_sync["timestamp"] - t_0 - 8.0, np.ones_like(self.df_sync["timestamp"]) * 20.0 * 0.1, c="black")
        plt.plot(self.df_sync["timestamp"] - t_0 - 8.0, np.ones_like(self.df_sync["timestamp"]) * 20.0 * 0.9, c="black")
        plt.xlim([0.0, 5.0])
        ticks = np.arange(0.0, 5.5, 0.5)
        plt.xlabel('Time (s)', fontsize=18)
        plt.xticks(ticks, fontsize=14)
        plt.ylabel('Pitch Angle (°)', fontsize=18)
        plt.ylim([-5.0, 25.0])
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, loc="upper left")


if __name__ == "__main__":
    figure_maker_roll = FigureMaker("experiments/feedback_response/231128_feedback_response_roll_arduino",
                                    "experiments/feedback_response/231128_feedback_response_roll_desired",
                                    "experiments/feedback_response/231128_feedback_response_roll_sync")
    # figure_maker_roll.file_synchronisation("experiments/feedback_response/231128_feedback_response_roll_sync")
    figure_maker_roll.plot_feedback_response_roll()

    figure_maker_pitch = FigureMaker("experiments/feedback_response/231128_feedback_response_pitch_arduino",
                                     "experiments/feedback_response/231128_feedback_response_pitch_desired",
                                     "experiments/feedback_response/231128_feedback_response_pitch_sync")
    # figure_maker_pitch.file_synchronisation("experiments/feedback_response/231128_feedback_response_pitch_sync")
    figure_maker_pitch.plot_feedback_response_pitch()
    plt.show()
