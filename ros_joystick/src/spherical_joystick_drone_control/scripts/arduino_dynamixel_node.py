#!/usr/bin/env python3

#THIS CODE RECEIVES ANGLE VALUES FROM SENSORS AND CONVERTS THEM INTO ROLL, PITCH AND YAW OF THE END-EFFECTOR USING THE FORWARD KINEMATICS
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import csv
import datetime

from arduino_serial_com import SerialCommunication
from serial.tools import list_ports

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, Float32


class ArduinoDynamixelNode:
    """
    This ROS node is used to communicate with the Arduino board and connect it to ROS.
    """
    def __init__(self):
        # get the port name automatically
        port_list = list(list_ports.comports(include_links=False))
        port_arduino = port_list[0].device
        print(port_arduino)  

        # Establish serial communication with Arduino
        self.ser_com = SerialCommunication(port_arduino, 115200)

        self.pub_joint_angles = rospy.Publisher('/dynamixel_joint_angles', Float32MultiArray, queue_size=1)
        self.pub_height_input = rospy.Publisher('/height_input', Float32, queue_size=1)

        self.joint_angles_msg = Float32MultiArray()
        self.height_input_msg = Float32()

        # Message to send to arduino
        self.joint_angles_des = np.zeros(5)

        # Data file for saving the data
        self.file_headers = ["datetime", "timestamp", 
                             "joint_angle_1", "joint_angle_1_des", 
                             "joint_angle_2", "joint_angle_2_des", 
                             "joint_angle_3", "joint_angle_3_des",
                             "joint_angle_4", "joint_angle_4_des",]

        self.file_name = "231128_feedback_response_pitch_arduino"
        
        # Initialize csv file with header
        with open(self.file_name, mode='w') as file:
            # Create a csv writer object
            writer = csv.writer(file, dialect="excel")
            writer.writerow(self.file_headers)

            file.close()
    
    # Extract joint angles from serial communication string
    def edit_data(self, data):
        # Split the data into the single parts
        split_data = data.split(",")

        # Split data string into joint angles
        joint_angle_1 = float(split_data[0][1:]) * (180 / np.pi)
        joint_angle_2 = float(split_data[2]) * (180 / np.pi)
        joint_angle_3 = float(split_data[4]) * (180 / np.pi)
        joint_angle_4 = float(split_data[6]) * (180 / np.pi)

        joint_angles = np.array((joint_angle_1, joint_angle_2, joint_angle_3, joint_angle_4))
        
        height_input = float(split_data[8][:-1])

        return joint_angles, height_input

    def send_desired_joint_angles_callback(self, msg):
        self.joint_angles_des = np.array([msg.data[0] * (np.pi / 180), msg.data[1] * (np.pi / 180), msg.data[2] * (np.pi / 180), msg.data[3] * (np.pi / 180), msg.data[4]])

    def main(self):
        rospy.init_node('arduino_dynamixel_node')
        rospy.Subscriber("/dynamixel_joint_angles_desired_feedback", Float32MultiArray, self.send_desired_joint_angles_callback)
        
        rate = rospy.Rate(100) # 100 Hz
        
        while not rospy.is_shutdown():
            # Run serial communication
            try:
                # Get data from serial
                data = self.ser_com.receive_data_from_arduino()
                joint_angles, height_input = self.edit_data(data) 

                # Define datetime and timestamp
                ct = datetime.datetime.now()
                ts = ct.timestamp()
                
                command = str(self.joint_angles_des[0]) + ' ' + str(self.joint_angles_des[1]) + ' ' + str(self.joint_angles_des[2]) + ' ' + str(self.joint_angles_des[3]) + ' ' + str(self.joint_angles_des[4]) + ' \r\n'
                print(command)
                self.ser_com.send_command_to_arduino(command)

                # Write to csv file
                with open(self.file_name, mode='a') as file:
                    # Create a csv writer object
                    writer = csv.writer(file, dialect="excel")

                    # Add row to csv file
                    writer.writerow([ct, ts, 
                                     joint_angles[0], self.joint_angles_des[0] * (180 / np.pi), 
                                     joint_angles[1], self.joint_angles_des[1] * (180 / np.pi), 
                                     joint_angles[2], self.joint_angles_des[2] * (180 / np.pi), 
                                     joint_angles[3], self.joint_angles_des[3] * (180 / np.pi)])

                    file.close()

            except KeyboardInterrupt:
                self.ser_com.ser.close()
        

            # Get joint angles and publish
            self.joint_angles_msg.data = joint_angles
            self.height_input_msg.data = height_input

            self.pub_joint_angles.publish(self.joint_angles_msg)
            self.pub_height_input.publish(self.height_input_msg)

            #rospy.loginfo(f"Joint angles: {self.joint_angles_msg.data}")

            rate.sleep()


if __name__ == '__main__':
    try:
        arduino_node = ArduinoDynamixelNode()
        arduino_node.main()
    except rospy.ROSInterruptException:
        pass

