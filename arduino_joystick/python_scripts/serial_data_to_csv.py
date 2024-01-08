import csv
import datetime
from serial.tools import list_ports

from arduino_serial_com import SerialCommunication


class SerialData2Csv:
    """
    Class for writing serial data to csv file.
    """

    def __init__(self, file_name):
        # get the port name automatically
        port_list = list(list_ports.comports(include_links=False))
        port_arduino = port_list[0].device
        print(port_arduino)  

        # Establish serial communication with Arduino
        self.ser_com = SerialCommunication(port_arduino)

        # Data frame for saving the data
        self.file_headers = ["datetime", "timestamp", "joint_angle_1", "joint_angle_des_1", "joint_angle_2", "joint_angle_des_2", "joint_angle_3", "joint_angle_des_3", "joint_angle_4", "joint_angle_des_4"]
        self.file_name = file_name

        # Initialize csv file with header
        with open(self.file_name, mode='w') as file:
            # Create a csv writer object
            writer = csv.writer(file, dialect="excel")
            writer.writerow(self.file_headers)

            file.close()

    # Extract pressure sensor readings and desired pressure from data string
    def edit_data(self, data):
        # Split the data into the single parts
        split_data = data.split(",")

        # Define datetime and timestamp
        ct = datetime.datetime.now()
        ts = ct.timestamp()

        # Split data string into translation and rotation data of base and end-effector
        joint_angle_1 = float(split_data[0][1:])
        joint_angle_des_1 = float(split_data[1])
        joint_angle_2 = float(split_data[2])
        joint_angle_des_2 = float(split_data[3])
        joint_angle_3 = float(split_data[4])
        joint_angle_des_3 = float(split_data[5])
        joint_angle_4 = float(split_data[6])
        joint_angle_des_4 = float(split_data[7][:-1])

        return ct, ts, joint_angle_1, joint_angle_des_1, joint_angle_2, joint_angle_des_2, joint_angle_3, joint_angle_des_3, joint_angle_4, joint_angle_des_4

    def run(self):
        # Run serial communication
        try:
            while True:
                # Get data from serial
                data = self.ser_com.receive_data_from_arduino()
                ct, ts, joint_angle_1, joint_angle_des_1, joint_angle_2, joint_angle_des_2, joint_angle_3, joint_angle_des_3, joint_angle_4, joint_angle_des_4 = self.edit_data(data)

                with open(self.file_name, mode='a') as file:
                    # Create a csv writer object
                    writer = csv.writer(file, dialect="excel")

                    # Add row to csv file
                    writer.writerow([ct, ts, joint_angle_1, joint_angle_des_1, joint_angle_2, joint_angle_des_2, joint_angle_3, joint_angle_des_3, joint_angle_4, joint_angle_des_4])

                    file.close()

                print(f"Joint angle 1: {joint_angle_1}, joint angle 1 desired: {joint_angle_des_1}, joint angle 2: {joint_angle_2}, joint angle 2 desired: {joint_angle_des_2}, joint angle 3: {joint_angle_3}, joint angle 3 desired: {joint_angle_des_3}, joint angle 4: {joint_angle_4}, joint angle 4 desired: {joint_angle_des_4}")

            print(f"Data saved to .csv file: {self.file_name}")

        except KeyboardInterrupt:
            self.ser_com.ser.close()


if __name__ == "__main__":
    # Start writing the serial data to csv serial communication
    data_writer = SerialData2Csv(file_name="date_name_raw.csv")
    data_writer.run()
