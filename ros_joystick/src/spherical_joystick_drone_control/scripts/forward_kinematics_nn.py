import torch
import torch.nn as nn


class ForwardKinematicsNN(nn.Module):
    """
    Feedforward neural network that takes motor joint angles as input
    and predicts end-effector roll, pitch and yaw angle.
    """

    def __init__(self, input_dim, output_dim, n_neurons=128):
        super(ForwardKinematicsNN, self).__init__()
        # Get parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons

        # Define fully-connected layers with ReLU activation
        self.fc_1 = nn.Linear(self.input_dim, self.n_neurons, dtype=torch.float32)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(self.n_neurons, self.n_neurons, dtype=torch.float32)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(self.n_neurons, self.n_neurons, dtype=torch.float32)
        self.relu_3 = nn.ReLU()
        self.fc_4 = nn.Linear(self.n_neurons, self.output_dim, dtype=torch.float32)

    def forward(self, x_in):
        """
        Parameters
        ----------
        x_in: joint angles

        Returns
        -------
        output: end-effector orientation (yaw, pitch, roll)
        """
        # Feedforward neural network with 4 fully-connected layers (2 hidden layers)
        x_fc1 = self.fc_1(x_in)
        x = self.relu_1(x_fc1)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        x = self.relu_3(x)
        x = self.fc_4(x)
        return x


if __name__ == "__main__":
    # Desired orientation of end-effector as [yaw, pitch, roll] in degrees
    desired_joint_angle = torch.tensor([90.09019470214844, 89.3296127319336, 89.86616516113281, 89.8099594116211], dtype=torch.float32)

    fk_nn = ForwardPredictionNN("models/231020_exp_4_spherical_joint_fk_ypr_best")
    yaw, pitch, roll = fk_nn(desired_joint_angle)
    print("--- FORWARD KINEMATICS RESULTS ---")
    print(f"End-effector orientation prediction: [{yaw}, {pitch}, {roll}]")