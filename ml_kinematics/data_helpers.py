import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def vicon_data_preparation(file_name):
    """
    Get the end-effector orientation data from vicon motion capturing data as .csv file.
    """

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
    quat_ee0 = np.array([-0.06710560247538873, -0.01691260236265429,
                         -0.33344333669962284, 0.9402267509533856])

    rot_ee0 = R.from_quat(quat_ee0)
    rot_base_ee0 = R.from_matrix(rot_base.inv().as_matrix() @ rot_ee0.as_matrix())

    # Get yaw, pitch, roll angles of the end-effector positions
    ypr_ee = np.array([rot.as_euler('zyx', degrees=True) for rot in rot_base_ee]) - rot_base_ee0.as_euler('zyx', degrees=True)

    return ypr_ee


def dynamixel_data_preparation(file_name):
    """
    Get the motor joint angle data from dynamixel motors as .csv file.
    """

    # Get joint angle data of dynamixel motors
    df = pd.read_csv(file_name)

    # Get joint angles
    joint_angles_rad = np.array([[df["joint_angle_1"][i],
                                  df["joint_angle_2"][i],
                                  df["joint_angle_3"][i],
                                  df["joint_angle_4"][i]] for i in range(len(df))])

    # Convert to degrees
    joint_angles_deg = joint_angles_rad * 180 / np.pi

    return joint_angles_deg


def file_synchronisation(file_name_1, file_name_2, file_name_save):
    """
    Synchronise the data from dynamixel .csv file and vicon .csv file using the timestamp.
    """

    print("Start File Synchronisation!")

    # Get joint angle data of dynamixel motors
    df_dynamixel = pd.read_csv(file_name_1)
    # Get end-effector position data of vicon system
    df_vicon = pd.read_csv(file_name_2)

    # Get timestamp at start and end of the files
    timestamp_start_dynamixel = df_dynamixel["timestamp"][0]
    timestamp_start_vicon = df_vicon["timestamp"][0]
    timestamp_end_dynamixel = df_dynamixel["timestamp"].iloc[-1]
    timestamp_end_vicon = df_vicon["timestamp"].iloc[-1]

    # Take the timestamp of the file that was started at a later (higher) timestamp
    if timestamp_start_dynamixel > timestamp_start_vicon:
        timestamp_start_target = timestamp_start_dynamixel
    else:
        timestamp_start_target = timestamp_start_vicon

    # Take the timestamp of the file that was finshed at a earlier (lower) timestep
    if timestamp_end_dynamixel > timestamp_end_vicon:
        timestamp_end_target = timestamp_end_vicon
    else:
        timestamp_end_target = timestamp_end_dynamixel

    for index, row in df_dynamixel.iterrows():
        if timestamp_end_target < row["timestamp"] < timestamp_start_target:
            df_dynamixel = df_dynamixel.drop(index)

    for index, row in df_vicon.iterrows():
        if timestamp_end_target < row["timestamp"] < timestamp_start_target:
            df_vicon.drop(index)

    df_dynamixel = df_dynamixel.reset_index()
    df_vicon = df_vicon.reset_index()

    """
    # --- ONLY FOR DATA SET ---
    # Filter dataframe: Difference of consecutive rows is zero (no motion) and only exist once (unique)
    idx_drop = []
    
    for idx, row in df_dynamixel.iterrows():
        if idx > 0:
            joint_angle_diff = (df_dynamixel["joint_angle_1"][idx] - df_dynamixel["joint_angle_1"][idx - 1])\
                               + (df_dynamixel["joint_angle_2"][idx] - df_dynamixel["joint_angle_2"][idx - 1])\
                               + (df_dynamixel["joint_angle_3"][idx] - df_dynamixel["joint_angle_3"][idx - 1])\
                               + (df_dynamixel["joint_angle_4"][idx] - df_dynamixel["joint_angle_4"][idx - 1])

            if joint_angle_diff > 0.0:
                idx_drop.append(idx)
    
    df_dynamixel = df_dynamixel.drop(idx_drop)
    """

    # Add the matching vicon data to the dynamixel dataframe
    df_dynamixel = df_dynamixel.reset_index()
    df_vicon = df_vicon.reset_index()
    vicon_timestamp = []
    # Orientation
    ee_base_ori_x = []
    ee_base_ori_y = []
    ee_base_ori_z = []
    ee_base_ori_w = []
    ee_top_ori_x = []
    ee_top_ori_y = []
    ee_top_ori_z = []
    ee_top_ori_w = []
    # Translation
    ee_base_pos_x = []
    ee_base_pos_y = []
    ee_base_pos_z = []
    ee_top_pos_x = []
    ee_top_pos_y = []
    ee_top_pos_z = []

    print(len(df_dynamixel))
    print(len(df_vicon))

    for dynamixel_idx, dynamixel_row in df_dynamixel.iterrows():
        print(f"Dynamixel file row: {dynamixel_idx}")

        time_diff = [np.abs(dynamixel_row["timestamp"] - vicon_row["timestamp"]) for vicon_idx, vicon_row in df_vicon.iterrows()]

        vicon_matching_idx = np.argmin(time_diff)
        vicon_timestamp.append(df_vicon["timestamp"][vicon_matching_idx])

        # orientation
        ee_base_ori_x.append(df_vicon["Spherical_Joint_Base_Ori_0"][vicon_matching_idx])
        ee_base_ori_y.append(df_vicon["Spherical_Joint_Base_Ori_1"][vicon_matching_idx])
        ee_base_ori_z.append(df_vicon["Spherical_Joint_Base_Ori_2"][vicon_matching_idx])
        ee_base_ori_w.append(df_vicon["Spherical_Joint_Base_Ori_3"][vicon_matching_idx])
        ee_top_ori_x.append(df_vicon["Spherical_Joint_Top_Ori_0"][vicon_matching_idx])
        ee_top_ori_y.append(df_vicon["Spherical_Joint_Top_Ori_1"][vicon_matching_idx])
        ee_top_ori_z.append(df_vicon["Spherical_Joint_Top_Ori_2"][vicon_matching_idx])
        ee_top_ori_w.append(df_vicon["Spherical_Joint_Top_Ori_3"][vicon_matching_idx])

        # translation
        ee_base_pos_x.append(df_vicon["Spherical_Joint_Base_Trans_0"][vicon_matching_idx])
        ee_base_pos_y.append(df_vicon["Spherical_Joint_Base_Trans_1"][vicon_matching_idx])
        ee_base_pos_z.append(df_vicon["Spherical_Joint_Base_Trans_2"][vicon_matching_idx])
        ee_top_pos_x.append(df_vicon["Spherical_Joint_Top_Trans_0"][vicon_matching_idx])
        ee_top_pos_y.append(df_vicon["Spherical_Joint_Top_Trans_1"][vicon_matching_idx])
        ee_top_pos_z.append(df_vicon["Spherical_Joint_Top_Trans_2"][vicon_matching_idx])

    # Add the vicon data to the dynamixel dataframe - Timestamp
    df_dynamixel["timestamp_vicon"] = pd.Series(vicon_timestamp)

    # Add the vicon data to the dynamixel dataframe - Orientation
    df_dynamixel["ee_base_ori_x"] = pd.Series(ee_base_ori_x)
    df_dynamixel["ee_base_ori_y"] = pd.Series(ee_base_ori_y)
    df_dynamixel["ee_base_ori_z"] = pd.Series(ee_base_ori_z)
    df_dynamixel["ee_base_ori_w"] = pd.Series(ee_base_ori_w)
    df_dynamixel["ee_top_ori_x"] = pd.Series(ee_top_ori_x)
    df_dynamixel["ee_top_ori_y"] = pd.Series(ee_top_ori_y)
    df_dynamixel["ee_top_ori_z"] = pd.Series(ee_top_ori_z)
    df_dynamixel["ee_top_ori_w"] = pd.Series(ee_top_ori_w)

    # Add the vicon data to the dynamixel dataframe - Translation
    df_dynamixel["ee_base_pos_x"] = pd.Series(ee_base_pos_x)
    df_dynamixel["ee_base_pos_y"] = pd.Series(ee_base_pos_y)
    df_dynamixel["ee_base_pos_z"] = pd.Series(ee_base_pos_z)
    df_dynamixel["ee_top_pos_x"] = pd.Series(ee_top_pos_x)
    df_dynamixel["ee_top_pos_y"] = pd.Series(ee_top_pos_y)
    df_dynamixel["ee_top_pos_z"] = pd.Series(ee_top_pos_z)

    # Save the synchronised dataframes to a .csv file
    df_dynamixel.to_csv(file_name_save, index=False)

    print("File Synchronisation Finished!")


def drop_duplicates(file_name, file_name_save):
    """
    Drop duplicate rows from dataframe.
    """

    df_sync = pd.read_csv(file_name)
    print(len(df_sync))
    df_sync = df_sync.drop_duplicates(["joint_angle_1", "joint_angle_2", "joint_angle_3", "joint_angle_4"])
    print(len(df_sync))
    df_sync.to_csv(file_name_save)


if __name__ == "__main__":
    # Possibility to drop duplicate rows from dataframe
    # drop_duplicates("data/231020_experiment_4_sync.csv", file_name_save="231020_experiment_4_sync_without_duplicates.csv")

    # Synchronise dynamixel and vicon data with input: dynamixel .csv file, vicon .csv file, file name to save synchronised data
    file_synchronisation("experiments/231024_trajectory_tracking_circle_40_degrees_1_dynamixel_raw.csv",
                         "experiments/231024_trajectory_tracking_circle_40_degrees_1_vicon_raw.csv",
                         file_name_save="experiments/231024_trajectory_tracking_circle_40_degrees_1_sync.csv")
