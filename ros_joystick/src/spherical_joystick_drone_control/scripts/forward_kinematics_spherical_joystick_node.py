#!/usr/bin/env python3

import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import numpy as np
from forward_kinematics_nn import ForwardKinematicsNN

import math
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, Float32


class ForwardKinematicsNode:
    """
    This ROS node is used to compute the drone control commands 
    based on calculating the end effector orientation (roll, pitch and yaw) 
    using the learned forward kinematics of the joystick.
    """

    def __init__(self):
        # Define forward kinematics neural network
        self.fk_nn_model_name = "/home/administrator/joystick_ws/src/spherical_joystick_drone_control/scripts/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(f'{self.fk_nn_model_name}'))
        self.fk_nn.eval()

        # Joint angle of dynamixel
        self.dynamixel_joint_angles = torch.zeros(4)
        self.height_input = 0
        self.joint_angle_min = 60
        self.joint_angle_max = 120

        # Normalization of FK output for drone input (-1: backward motion, 0: no motion, 1: forward motion)
        self.threshold = 0.2
        self.roll_norm = 50.0
        self.pitch_norm = 50.0
        self.yaw_norm = 50.0
        self.pose_msg = Pose()

    def callback_joint_angles(self, msg):
        self.dynamixel_joint_angles[0] = msg.data[0]
        self.dynamixel_joint_angles[1] = msg.data[1]
        self.dynamixel_joint_angles[2] = msg.data[2]
        self.dynamixel_joint_angles[3] = msg.data[3]
    
    def callback_height_input(self, msg):
        self.height_input = msg.data

    def main(self):
        rospy.init_node('forward_kinematics_spherical_joystick_node')
        rospy.Subscriber("/dynamixel_joint_angles", Float32MultiArray, self.callback_joint_angles)
        rospy.Subscriber("/height_input", Float32, self.callback_height_input)
        pub = rospy.Publisher('/drone_control', Pose, queue_size=1)
        rate = rospy.Rate(100) # 100Hz         

        while not rospy.is_shutdown():
            # compute the FK with the neural network
            with torch.no_grad():
                joint_angle_configuration_norm = (self.dynamixel_joint_angles - self.joint_angle_min) / (self.joint_angle_max - self.joint_angle_min)
                torch_yaw_pitch_roll_pred = self.fk_nn(joint_angle_configuration_norm) 
                yaw_pitch_roll_pred = torch_yaw_pitch_roll_pred.detach().numpy()

            # Calibrate Raw/Pitch/Norm values into game controls 
            motion_forward_backward = np.clip(yaw_pitch_roll_pred[1] / self.pitch_norm, -1.0, 1.0)
            motion_left_right = np.clip(yaw_pitch_roll_pred[2] / self.roll_norm, -1.0, 1.0)
            motion_rotate = np.clip(yaw_pitch_roll_pred[0] / self.yaw_norm, -1.0, 1.0)

            # position.z in Unity
            if np.abs(self.height_input) < self.threshold:
                self.pose_msg.position.x = 0
            else:
                self.pose_msg.position.x = np.clip(-self.height_input*0.5, -1.0, 1.0)

            # position.x in Unity
            if np.abs(motion_forward_backward) < self.threshold:
                self.pose_msg.position.y = 0
            else:
                self.pose_msg.position.y = -motion_forward_backward
            
            # position.y in Unity
            if np.abs(motion_left_right) < self.threshold:
                self.pose_msg.position.z = 0
            else:
                self.pose_msg.position.z = motion_left_right  
            
            # rotation in Unity
            if np.abs(motion_rotate) < self.threshold:
                self.pose_msg.orientation.x = 0
            else:
                self.pose_msg.orientation.x = -motion_rotate

            pub.publish(self.pose_msg)

            rospy.loginfo(f"Drone position cmd: {self.pose_msg.position}")
            rospy.loginfo(f"Drone orienation cmd: {self.pose_msg.orientation}")

            rate.sleep()


if __name__ == '__main__':
    try:
        fk_node = ForwardKinematicsNode()
        fk_node.main()
    except rospy.ROSInterruptException:
        pass
