#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import torch
import datetime
import csv

from forward_kinematics_nn import ForwardKinematicsNN
from inverse_kinematics_cem import cem_based_optimization


class ContactFeedbackNode:
    """
    This ROS node is used to calculate the desired joint angles for the haptic feedback.
    """

    def __init__(self):
        # Define if simulation is used or step response is given
        self.simulation = False
        self.roll_step = False # Otherwise pitch step
        self.step_input = 20.0
        self.i = 0 # counter
        self.step_counter = 500 # 500 - 5 seconds with frequency of 100

        # Collision data from unity
        self.previous_contact = False
        self.actual_contact = False
        self.publish = False
        self.contact = 0.0
        self.contact_msg = np.zeros(3)

        # Desired joint angles for haptic feedback
        self.dynamixel_joint_angles_des_msg = Float32MultiArray()
        self.dynamixel_joint_angles_des_msg.data = np.array([120, 60, 120, 60, 10]) # Joint angles and motor current
        self.roll_norm = 100.0
        self.pitch_norm = 100.0

        # Inverse Kinematics for haptic feedback
        self.fk_nn_model_name = "/home/administrator/joystick_ws/src/spherical_joystick_drone_control/scripts/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(self.fk_nn_model_name))
        self.fk_nn.eval()
        # Cross entropy method (CEM)
        self.n_samples = 1000
        self.num_opti_iters = 10
        self.num_elites = 100

        # Data file for saving the data
        self.file_headers = ["counter", "datetime", "timestamp", "yaw_des", "pitch_des", "roll_des"]
        self.file_name = "231128_feedback_response_pitch_desired"
        
        # Initialize csv file with header
        with open(self.file_name, mode='w') as file:
            # Create a csv writer object
            writer = csv.writer(file, dialect="excel")
            writer.writerow(self.file_headers)

            file.close()
    
    def callback_collsion(self, msg):
        # Check if contact occured
        #self.contact = msg.pose.orientation.y
        self.contact = msg.pose.position.x

        # unity changes the x,y,z coordinates, values between -1 and +1
        self.contact_msg[0] = msg.pose.position.x
        self.contact_msg[1] = msg.pose.position.y
        self.contact_msg[2] = msg.pose.position.z

    def main(self):
        rospy.init_node('contact_feedback_node')
        rospy.Subscriber("/collision", PoseStamped, self.callback_collsion)
        pub = rospy.Publisher('/dynamixel_joint_angles_desired_feedback', Float32MultiArray, queue_size=1)
        rate = rospy.Rate(100) # 100Hz
        
        while not rospy.is_shutdown():
            if self.simulation:
                if self.contact != 0:
                    # Transform contact message from x,y,z into [yaw, pitch, roll] angles
                    yaw = 0.0
                    pitch = -self.contact_msg[0] * self.pitch_norm
                    roll = self.contact_msg[1] * self.roll_norm
                    joystick_ori = torch.tensor([yaw, pitch, roll])

                    # Calculate the joint angles by using the inverse kinematics
                    ik_joint_angles, _, _ = cem_based_optimization(self.fk_nn, joystick_ori, self.n_samples, self.num_opti_iters, self.num_elites)
                    motor_current = 50
                    
                    # Define data to be published
                    print(ik_joint_angles)
                    data = np.append(ik_joint_angles, motor_current)
                    self.dynamixel_joint_angles_des_msg.data = data

                else:
                    # Joint angles of resting position of joystick
                    self.dynamixel_joint_angles_des_msg.data = np.array([120, 60, 120, 60, 10])
            
            else:
                if self.i < self.step_counter:
                    yaw = 0.0
                    pitch = 0.0
                    roll = 0.0
                
                else:
                    if self.roll_step:
                        yaw = 0.0
                        pitch = 0.0
                        roll = self.step_input
                    else:
                        yaw = 0.0
                        pitch = self.step_input
                        roll = 0.0
                
                motor_current = 150
                
                # Define datetime and timestamp
                ct = datetime.datetime.now()
                ts = ct.timestamp()
                
                # Write to csv file
                with open(self.file_name, mode='a') as file:
                    # Create a csv writer object
                    writer = csv.writer(file, dialect="excel")

                    # Add row to csv file
                    writer.writerow([self.i, ct, ts, yaw, pitch, roll])

                    file.close()
                    
                joystick_ori = torch.tensor([yaw, pitch, roll])

                # Calculate the joint angles by using the inverse kinematics
                ik_joint_angles, _, _ = cem_based_optimization(self.fk_nn, joystick_ori, self.n_samples, self.num_opti_iters, self.num_elites)
                    
                # Define data to be published
                data = np.append(ik_joint_angles, motor_current)
                self.dynamixel_joint_angles_des_msg.data = data
                
                print(self.i)
                self.i += 1
                

            pub.publish(self.dynamixel_joint_angles_des_msg)
            print(self.dynamixel_joint_angles_des_msg.data)

            rate.sleep()


if __name__ == '__main__':
    try:
        contact_node = ContactFeedbackNode()
        contact_node.main()
    except rospy.ROSInterruptException:
        pass
