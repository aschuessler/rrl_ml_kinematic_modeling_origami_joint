# Machine Learning-based Kinematic Modeling of a Multi-loop Origami Spherical Joint with Hidden Degrees of Freedom

This is the repository for the paper "Machine Learning-based Kinematic Modeling of a Multi-loop Origami Spherical Joint with Hidden Degrees of Freedom" by Mete et al..
The repository includes folders with the method implementation, data set and experimental data, as well as the Arduino and ROS code for executing the methods as a drone joystick.
DOI for Release v1.0: [https://doi.org/10.5281/zenodo.10470770](https://doi.org/10.5281/zenodo.10470770)

## arduino_joystick
This folder contains the Arduino code for controlling the Dynamixel motors of the joystick.

## ml_kinematics
This folder contains the machine learning algorithms and models used for kinematic modeling of the origami joint. It includes the data set processing, model training, and experiment scripts.

## ros_joystick
This folder contains the ROS (Robot Operating System) package for interfacing the Arduino joystick with the ROS environment. 
This allows for integration with Unity drone simulation through ROS.

## License
This project is licensed under the GPL-3.0 License.