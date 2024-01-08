import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class FigureMaker:
    def __init__(self, file_name_recording, file_name_target, target_degree, type, circle_3d_plot=False):
        # Open csv file as pandas dataframe
        self.file_name_recording = file_name_recording
        self.file_name_target = file_name_target
        self.df_record = pd.read_csv(self.file_name_recording)
        self.df_target = pd.read_csv(self.file_name_target)

        # Define physical parameters of joystick
        self.joystick_length = 80.0  # in mm
        self.target_degree = target_degree
        self.joystick_top_radius = 15.0

        self.plot_type = type
        self.circle_3d_plot = circle_3d_plot

    def vicon_data_preparation(self, file_name):
        # Get end-effector data from vicon
        df = pd.read_csv(file_name)

        # Get base frame
        quat_base = np.array([df["ee_base_ori_x"][0],
                              df["ee_base_ori_y"][0],
                              df["ee_base_ori_z"][0],
                              df["ee_base_ori_w"][0]])
        rot_base = R.from_quat(quat_base)

        # Get end-effector (ee) frame
        quat_ee = np.array([[df["ee_top_ori_x"][i],
                             df["ee_top_ori_y"][i],
                             df["ee_top_ori_z"][i],
                             df["ee_top_ori_w"][i]] for i in range(len(df))])
        rot_ee = np.array([R.from_quat(quat) for quat in quat_ee])

        # Get transformation matrix from base to end-effector
        rot_base_ee = np.array([R.from_matrix(rot_base.inv().as_matrix() @ rot.as_matrix()) for rot in rot_ee])

        # Get transformation matrix from end-effector position of first sample (ideally initial position)
        quat_ee0 = quat_ee[0, :]
        rot_ee0 = R.from_quat(quat_ee0)
        rot_base_ee0 = R.from_matrix(rot_base.inv().as_matrix() @ rot_ee0.as_matrix())

        # Get yaw, pitch, roll angles of the end-effector positions
        ypr_ee = np.array([rot.as_euler('zyx', degrees=True) for rot in rot_base_ee]) - rot_base_ee0.as_euler('zyx', degrees=True)

        # Get quaternion of the end-effector positions
        quat_ee = np.array([rot.as_quat() for rot in rot_base_ee])

        # Get position of end-effector relative to the starting position
        pos_ee = np.array([[df["ee_top_pos_x"][i],
                            df["ee_top_pos_y"][i],
                            df["ee_top_pos_z"][i]] for i in range(len(df))])
        pos_ee0 = np.array([df["ee_top_pos_x"][0], df["ee_top_pos_y"][0], df["ee_top_pos_z"][0]])
        pos_base_ee = np.array([rot_base.inv().as_matrix() @ (pos - pos_ee0) for pos in pos_ee]) + np.array([0.0, 0.0, self.joystick_length])
        pos_base_ee_side = np.array([rot_base.inv().as_matrix() @ (pos - pos_ee0 + np.array([15.0, 0.0, 0.0])) for pos in pos_ee]) + np.array([0.0, 0.0, self.joystick_length])

        return ypr_ee, quat_ee, pos_base_ee, pos_base_ee_side

    def plot_step_series(self, save_figure_name):
        # Data preparation
        ypr_ee, quat_ee, _, _ = self.vicon_data_preparation(self.file_name_recording)
        t_0 = self.df_record["timestamp"][1]

        # Create a figure with three subplots stacked vertically (roll, pitch and yaw angle)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=False)

        # Plot for roll angle
        if self.plot_type == "circle":
            target_traj = np.concatenate((self.df_target["target_roll_angle"].to_numpy(), np.zeros(10)))
            ax1.plot(self.df_record["timestamp"][1:41] - t_0, ypr_ee[1:41, 2], c="black", label="Circular trajectory")
            ax1.plot(self.df_record["timestamp"][1:41] - t_0, target_traj, c="black", linestyle="dashed", label="Target")
            ax1.set_xlim([0.0, 8.0])

            ax1.set_ylabel('Roll / °')
            ax1.set_ylim([-50.0, 50.0])
            ax1.legend()

            angular_accuracy_roll = np.mean(np.abs(ypr_ee[1:41, 2] - target_traj))

            target_traj = np.concatenate((self.df_target["target_pitch_angle"].to_numpy(), np.zeros(10)))
            ax2.plot(self.df_record["timestamp"][1:41] - t_0, ypr_ee[1:41, 1], c="black")
            ax2.plot(self.df_record["timestamp"][1:41] - t_0, target_traj, c="black", linestyle="dashed")
            ax2.set_xlim([0.0, 8.0])

            ax2.set_ylabel('Pitch / °')
            ax2.set_ylim([-50.0, 50.0])

            angular_accuracy_pitch = np.mean(np.abs(ypr_ee[1:41, 1] - target_traj))

            target_traj = np.concatenate((self.df_target["target_yaw_angle"].to_numpy(), np.zeros(10)))
            ax3.plot(self.df_record["timestamp"][1:41] - t_0, ypr_ee[1:41, 0], c="black")
            ax3.plot(self.df_record["timestamp"][1:41] - t_0, target_traj, c="black", linestyle="dashed")
            ax3.set_xlim([0.0, 8.0])

            angular_accuracy_yaw = np.mean(np.abs(ypr_ee[1:41, 0] - target_traj))

            angular_accuracy = (angular_accuracy_roll + angular_accuracy_pitch + angular_accuracy_yaw) / 3

            ax3.set_xlabel('Time / s')
            ax3.set_ylabel('Yaw / °')
            ax3.set_ylim([-50.0, 50.0])
            fig.savefig(save_figure_name, dpi=600)

            print("- Circular trajectory -")
            print(f"Mean angular accuracy roll: {angular_accuracy_roll:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_pitch:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_yaw:.4f}")
            print(f"Mean angular accuracy total: {angular_accuracy:.4f}")

        elif self.plot_type == "roll":
            target_traj = np.concatenate((self.df_target["target_roll_angle"].to_numpy(), np.zeros(10)))
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 2], c="black", label="Roll trajectory")
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed", label="Target")
            ax1.set_xlim([0.0, 5.0])

            ax1.set_ylabel('Roll / °')
            ax1.set_ylim([-50.0, 50.0])
            ax1.legend()

            angular_accuracy_roll = np.mean(np.abs(ypr_ee[1:31, 2] - target_traj))

            target_traj = np.concatenate((self.df_target["target_pitch_angle"].to_numpy(), np.zeros(10)))
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 1], c="black")
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax2.set_xlim([0.0, 5.0])

            ax2.set_ylabel('Pitch / °')
            ax2.set_ylim([-50.0, 50.0])

            angular_accuracy_pitch = np.mean(np.abs(ypr_ee[1:31, 1] - target_traj))

            target_traj = np.concatenate((self.df_target["target_yaw_angle"].to_numpy(), np.zeros(10)))
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 0], c="black")
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax3.set_xlim([0.0, 5.0])

            ax3.set_xlabel('Time / s')
            ax3.set_ylabel('Yaw / °')
            ax3.set_ylim([-50.0, 50.0])
            fig.savefig(save_figure_name, dpi=600)

            angular_accuracy_yaw = np.mean(np.abs(ypr_ee[1:31, 0] - target_traj))

            angular_accuracy = (angular_accuracy_roll + angular_accuracy_pitch + angular_accuracy_yaw) / 3

            print("- Roll trajectory -")
            print(f"Mean angular accuracy roll: {angular_accuracy_roll:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_pitch:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_yaw:.4f}")
            print(f"Mean angular accuracy total: {angular_accuracy:.4f}")

        elif self.plot_type == "pitch":
            target_traj = np.concatenate((self.df_target["target_roll_angle"].to_numpy(), np.zeros(10)))
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 2], c="black", label="Pitch trajectory")
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed", label="Target")
            ax1.set_xlim([0.0, 5.0])

            ax1.set_ylabel('Roll / °')
            ax1.set_ylim([-50.0, 50.0])
            ax1.legend()

            angular_accuracy_roll = np.mean(np.abs(ypr_ee[1:31, 2] - target_traj))

            target_traj = np.concatenate((self.df_target["target_pitch_angle"].to_numpy(), np.zeros(10)))
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 1], c="black")
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax2.set_xlim([0.0, 5.0])

            ax2.set_ylabel('Pitch / °')
            ax2.set_ylim([-50.0, 50.0])

            angular_accuracy_pitch = np.mean(np.abs(ypr_ee[1:31, 1] - target_traj))

            target_traj = np.concatenate((self.df_target["target_yaw_angle"].to_numpy(), np.zeros(10)))
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 0], c="black")
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax3.set_xlim([0.0, 5.0])

            ax3.set_xlabel('Time / s')
            ax3.set_ylabel('Yaw / °')
            ax3.set_ylim([-50.0, 50.0])
            fig.savefig(save_figure_name, dpi=600)

            angular_accuracy_yaw = np.mean(np.abs(ypr_ee[1:31, 0] - target_traj))

            angular_accuracy = (angular_accuracy_roll + angular_accuracy_pitch + angular_accuracy_yaw) / 3

            print("- Pitch trajectory -")
            print(f"Mean angular accuracy roll: {angular_accuracy_roll:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_pitch:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_yaw:.4f}")
            print(f"Mean angular accuracy total: {angular_accuracy:.4f}")

        elif self.plot_type == "yaw":
            target_traj = np.concatenate((self.df_target["target_roll_angle"].to_numpy(), np.zeros(10)))
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 2], c="black", label="Yaw trajectory")
            ax1.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed", label="Target")
            ax1.set_xlim([0.0, 5.0])

            ax1.set_ylabel('Roll / °')
            ax1.set_ylim([-50.0, 50.0])
            ax1.legend()

            angular_accuracy_roll = np.mean(np.abs(ypr_ee[1:31, 2] - target_traj))

            target_traj = np.concatenate((self.df_target["target_pitch_angle"].to_numpy(), np.zeros(10)))
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 1], c="black")
            ax2.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax2.set_xlim([0.0, 5.0])

            ax2.set_ylabel('Pitch / °')
            ax2.set_ylim([-50.0, 50.0])

            angular_accuracy_pitch = np.mean(np.abs(ypr_ee[1:31, 1] - target_traj))

            target_traj = np.concatenate((self.df_target["target_yaw_angle"].to_numpy(), np.zeros(10)))
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, ypr_ee[1:31, 0], c="black")
            ax3.plot(self.df_record["timestamp"][1:31] - t_0, target_traj, c="black", linestyle="dashed")
            ax3.set_xlim([0.0, 5.0])

            ax3.set_xlabel('Time / s')
            ax3.set_ylabel('Yaw / °')
            ax3.set_ylim([-50.0, 50.0])
            fig.savefig(save_figure_name, dpi=600)

            angular_accuracy_yaw = np.mean(np.abs(ypr_ee[1:31, 0] - target_traj))

            angular_accuracy = (angular_accuracy_roll + angular_accuracy_pitch + angular_accuracy_yaw) / 3

            print("- Yaw trajectory -")
            print(f"Mean angular accuracy roll: {angular_accuracy_roll:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_pitch:.4f}")
            print(f"Mean angular accuracy yaw: {angular_accuracy_yaw:.4f}")
            print(f"Mean angular accuracy total: {angular_accuracy:.4f}")

        else:
            print("Wrong plot type chosen. This can be roll, pitch, yaw or circle.")
            return

    def plot_3d(self, save_figure_name):
        # Data preparation
        ypr_ee, _, pos_ee, pos_ee_side = self.vicon_data_preparation(self.file_name_recording)

        if self.plot_type == "circle":
            # Positions for circle with 20 degrees of roll and pitch
            circle_radius = self.joystick_length * np.sin(np.pi * self.target_degree / 180)
            x_target = np.linspace(-circle_radius, circle_radius, 100)
            y_target_1 = np.array([np.sqrt(circle_radius ** 2 - x ** 2) for x in x_target])
            target_1 = np.array([x_target, y_target_1, np.zeros(100) + self.joystick_length])
            y_target_2 = np.array([- np.sqrt(circle_radius ** 2 - x ** 2) for x in x_target])
            target_2 = np.array([x_target, y_target_2, np.zeros(100) + self.joystick_length])
            target_3 = np.array([np.linspace(0.0, circle_radius, 100), np.zeros(100), np.zeros(100) + self.joystick_length])

            # Create a 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=22.5, azim=-45.0, roll=0.0)

            # Plot the trajectory of circle - 20 degrees
            ax.plot(pos_ee[:, 0], pos_ee[:, 1], pos_ee[:, 2], color="black", label='Circular actual')
            ax.plot(target_1[0, :], target_1[1, :], target_1[2, :], color="black", linestyle="dashed", label='Circular target')
            ax.plot(target_2[0, :], target_2[1, :], target_2[2, :], color="black", linestyle="dashed")
            ax.plot(target_3[0, :], target_3[1, :], target_3[2, :], color="black", linestyle="dashed")

            # Get rid of the panes
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Add labels
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('X (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim([-50.0, 50.0])
            ax.set_ylim([-50.0, 50.0])
            ax.set_zlim([0.0, 100.0])
            ax.legend(loc="upper right")
            fig.savefig(save_figure_name, dpi=600)

        elif self.plot_type == "roll":
            # Create a 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=22.5, azim=-45.0, roll=0.0)

            target = np.array([np.zeros(10), -self.joystick_length * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180), self.joystick_length * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180)])
            # Plot the trajectory
            ax.plot(pos_ee[:, 0], pos_ee[:, 1], pos_ee[:, 2], color="black", label='Roll trajectory')
            ax.plot(target[0, :], target[1, :], target[2, :], color="black", linestyle="dashed", label='Target')

            # Get rid of the panes
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Add labels
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('X (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim([-50.0, 50.0])
            ax.set_ylim([-50.0, 50.0])
            ax.set_zlim([0.0, 100.0])
            ax.legend()
            fig.savefig(save_figure_name, dpi=600)

        elif self.plot_type == "pitch":
            # Create a 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=22.5, azim=-45.0, roll=0.0)

            target = np.array([-self.joystick_length * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180), np.zeros(10), self.joystick_length * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180)])
            # Plot the trajectory
            ax.plot(pos_ee[:, 0], pos_ee[:, 1], pos_ee[:, 2], color="black", label='Pitch trajectory')
            ax.plot(target[0, :], target[1, :], target[2, :], color="black", linestyle="dashed", label='Target')

            # Get rid of the panes
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Add labels
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('X (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim([-50.0, 50.0])
            ax.set_ylim([-50.0, 50.0])
            ax.set_zlim([0.0, 100.0])
            ax.legend()
            fig.savefig(save_figure_name, dpi=600)

        elif self.plot_type == "yaw":
            # Create a 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=22.5, azim=-45.0, roll=0.0)

            # This is wrong
            target_1 = np.array([self.joystick_top_radius * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 11) / 180), self.joystick_top_radius * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 11) / 180), np.zeros(11) + self.joystick_length])
            target_2 = np.array([-self.joystick_top_radius * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 11) / 180), -self.joystick_top_radius * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 11) / 180), np.zeros(11) + self.joystick_length])
            actual_1 = np.array([self.joystick_top_radius * (np.cos(np.pi * ypr_ee[:, 0] / 180)), self.joystick_top_radius * (np.sin(np.pi * ypr_ee[:, 0] / 180)), np.zeros(len(ypr_ee[:, 0])) + self.joystick_length])
            actual_2 = np.array([-self.joystick_top_radius * (np.cos(np.pi * ypr_ee[:, 0] / 180)), -self.joystick_top_radius * (np.sin(np.pi * ypr_ee[:, 0] / 180)), np.zeros(len(ypr_ee[:, 0])) + self.joystick_length])

            ax.plot(actual_1[0, :], actual_1[1, :], actual_1[2, :], color="black", label='Actual Rotation')
            ax.plot(actual_2[0, :], actual_2[1, :], actual_2[2, :], color="black")
            ax.scatter(target_1[0, 5], target_1[1, 5], target_1[2, 5], color="blue", label='Target Rotation')
            ax.scatter(target_1[0, 0], target_1[1, 0], target_1[2, 0], color="blue")
            ax.scatter(target_1[0, -1], target_1[1, -1], target_1[2, -1], color="blue")
            ax.scatter(target_2[0, 0], target_2[1, 0], target_2[2, 0], color="red", label='Target Rotation')
            ax.scatter(target_2[0, 5], target_2[1, 5], target_2[2, 5], color="red")
            ax.scatter(target_2[0, -1], target_2[1, -1], target_2[2, -1], color="red")

            # Get rid of the panes
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Add labels
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('X (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim([-15.0, 15.0])
            ax.set_ylim([-15.0, 15.0])
            ax.set_zlim([0.0, 100.0])
            ax.legend(loc="best")
            fig.savefig(save_figure_name, dpi=600)

        else:
            print("NO 3D PLOT SELECTED")
            return

    def plot_3d_with_x(self, file_name_recording_x, file_name_target_x, save_figure_name):
        df_record_x = pd.read_csv(file_name_recording_x)
        df_target_x = pd.read_csv(file_name_target_x)

        # Data preparation
        ypr_ee, _, pos_ee, pos_ee_side = self.vicon_data_preparation(self.file_name_recording)

        # Data preparation of x
        ypr_ee_x, _, pos_ee_x, pos_ee_side_x = self.vicon_data_preparation(file_name_recording_x)

        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=22.5, azim=-45.0, roll=0.0)

        # Plot the trajectory of x - roll
        target_x = np.array([np.zeros(10), -self.joystick_length * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180), self.joystick_length * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180)])
        ax.plot(pos_ee_x[:, 0], pos_ee_x[:, 1], pos_ee_x[:, 2], color="darkred", label='Roll actual')
        ax.plot(target_x[0, :], target_x[1, :], target_x[2, :], color="darkred", linestyle="dashed", label='Roll target')

        # Plot the trajectory - pitch
        target = np.array([-self.joystick_length * np.sin(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180), np.zeros(10), self.joystick_length * np.cos(np.pi * np.linspace(-self.target_degree, self.target_degree, 10) / 180)])
        ax.plot(pos_ee[:, 0], pos_ee[:, 1], pos_ee[:, 2], color="mediumblue", label='Pitch actual')
        ax.plot(target[0, :], target[1, :], target[2, :], color="mediumblue", linestyle="dashed", label='Pitch target')

        # Get rid of the panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Add labels
        ax.set_xlabel('Y (mm)')
        ax.set_ylabel('X (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_xlim([-50.0, 50.0])
        ax.set_ylim([-50.0, 50.0])
        ax.set_zlim([0.0, 100.0])
        ax.legend()
        fig.savefig(save_figure_name, dpi=600)


if __name__ == "__main__":
    # Trajectory tracking - Roll
    fig_roll = FigureMaker("experiments/trajectory_tracking/231024_trajectory_tracking_roll_40_degrees_1_sync.csv",
                           "experiments/trajectory_tracking/231023_trajectory_tracking_20_samples_40_degrees_no_constrain_roll.csv", type="roll", target_degree=40)
    # fig_roll.plot_step_series("figures/fig_traj_time_roll_40_v1.png")
    # fig_roll.plot_3d("figures/fig_traj_3d_roll_40_v1.png")

    # Trajectory tracking - Pitch
    fig_pitch = FigureMaker("experiments/trajectory_tracking/231024_trajectory_tracking_pitch_40_degrees_1_sync.csv",
                            "experiments/trajectory_tracking/231023_trajectory_tracking_20_samples_40_degrees_no_constrain_pitch.csv", type="pitch", target_degree=40)
    # fig_pitch.plot_step_series("figures/fig_traj_time_pitch_40_v1.png")
    # fig_pitch.plot_3d("figures/fig_traj_3d_pitch_40_v1.png")
    fig_pitch.plot_3d_with_x("experiments/trajectory_tracking/231024_trajectory_tracking_roll_40_degrees_1_sync.csv",
                             "experiments/trajectory_tracking/231023_trajectory_tracking_20_samples_40_degrees_no_constrain_roll.csv",
                             "figures/fig_traj_3d_roll_pitch_40_v1.png")

    # Trajectory tracking - Yaw
    fig_yaw = FigureMaker("experiments/trajectory_tracking/231024_trajectory_tracking_yaw_15_degrees_1_sync.csv",
                          "experiments/trajectory_tracking/231024_trajectory_tracking_20_samples_15_degrees_no_constrain_yaw.csv", type="yaw", target_degree=15)
    # fig_yaw.plot_step_series("figures/fig_traj_time_yaw_15_v1.png")
    # fig_yaw.plot_3d("figures/fig_traj_3d_yaw_15_v1.png")

    # Trajectory tracking - Circle 20 degrees
    fig_circle_20 = FigureMaker("experiments/trajectory_tracking/231024_trajectory_tracking_circle_20_degrees_1_sync.csv",
                                "experiments/trajectory_tracking/231023_trajectory_tracking_20_samples_20_degrees_no_constrain_circle.csv", type="circle", target_degree=20, circle_3d_plot=True)
    # fig_circle_20.plot_step_series("figures/fig_traj_time_circle_20_v1.png")
    fig_circle_20.plot_3d("figures/fig_traj_3d_circle_20_v1.png")

    plt.show()
