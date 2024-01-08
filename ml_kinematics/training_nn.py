import torch
import torch.nn as nn
import torch.utils.data
import time
import pandas as pd
import wandb

from forward_kinematics_nn import ForwardKinematicsNN
from dataset import ForwardKinematicsDataset


class ForwardKinematicsNNTrainer:
    """
    Class for the training of the Forward Kinematics Neural Network
    """
    def __init__(self, num_epochs, n_neurons, learning_rate, l2_regu):
        # Define seed for reproducible results
        torch.manual_seed(10)

        # Define important parameters
        self.file_name = "data/231020_experiment_4_sync.csv"
        self.num_epochs = num_epochs
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.optimizer_weight_decay = l2_regu
        self.save_model_name = "231021_exp_4_spherical_joint_fk_ypr_test"
        self.save_model_dir = f"models/{self.save_model_name}"

        # Get datasets for training, validation and testing
        self.train_dataset = ForwardKinematicsDataset(self.file_name, norm=True, train=True)
        self.val_dataset = ForwardKinematicsDataset(self.file_name, norm=True, train=False)

        self.n_train_set = len(self.train_dataset)
        self.n_val_set = len(self.val_dataset)

        # Define manipulation prediction model
        self.forward_kinematics_model = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=self.n_neurons)

        # Loss and optimizer
        self.mse_loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.forward_kinematics_model.parameters(), lr=self.learning_rate, weight_decay=self.optimizer_weight_decay)

    def train(self):
        """
        Method that defines the training steps
        """
        print("--- Start Training Loop ---")
        overall_start_time = time.time()
        list_mean_loss = []
        list_mean_val_loss = []
        # Loop over epochs
        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Training step
            acc_loss = 0.0
            for i, (joint_angles, ee_ori_gt) in enumerate(self.train_dataset):
                # Forward pass
                ee_ori_pred = self.forward_kinematics_model(joint_angles)
                loss = self.mse_loss_function(ee_ori_pred, ee_ori_gt)
                acc_loss += loss.item()

                # Backward and optimize model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Define mean loss of training step
            mean_loss = acc_loss / self.n_train_set
            list_mean_loss.append(mean_loss)

            # Validation step, without gradient
            with torch.no_grad():
                acc_loss_val = 0.0
                accuracy_val = 0.0
                for i, (joint_angles, ee_ori_gt) in enumerate(self.val_dataset):
                    # Forward pass
                    ee_ori_pred = self.forward_kinematics_model(joint_angles)
                    loss_val = self.mse_loss_function(ee_ori_pred, ee_ori_gt)
                    accuracy_val += torch.sum(torch.abs(ee_ori_gt - ee_ori_pred)).item()
                    acc_loss_val += loss_val.item()

            # Define mean loss of validation step
            mean_val_loss = acc_loss_val / self.n_val_set
            mean_val_accuracy = accuracy_val / self.n_val_set
            list_mean_val_loss.append(mean_val_loss)

            # Print results of epoch
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]: '
                  f'Time {(time.time()-start_time):.1f}, '
                  f'Mean train loss: {mean_loss:.6f}, '
                  f'Mean validation loss: {mean_val_loss:.6f}')

            # wandb.log({"Mean train loss": mean_loss, "Mean validation loss": mean_val_loss, "Mean validation accuracy": mean_val_accuracy})

        # Save trained model
        torch.save(self.forward_kinematics_model.state_dict(), f'{self.save_model_dir}')

        # Save training data as .csv file
        overall_time = time.time() - overall_start_time
        training_data = {"mean_mse_loss": pd.Series(list_mean_loss),
                         "mean_mse_loss_val": pd.Series(list_mean_val_loss),
                         "overall_time": pd.Series(overall_time)}
        df = pd.DataFrame(training_data)
        df.to_csv(f'training_results/{self.save_model_name}.csv')

    def test(self):
        """
        Method that defines the test steps
        """
        print("--- Start Test Loop ---")
        # Test step, without gradient
        with torch.no_grad():
            acc_test_loss = 0.0
            acc_test_accuracy = torch.tensor([0.0, 0.0, 0.0])

            for i, (joint_angles, ee_ori_gt) in enumerate(self.val_dataset):
                # Forward pass
                prediction = self.forward_kinematics_model(joint_angles)
                loss = self.mse_loss_function(prediction, ee_ori_gt)
                accuracy = torch.abs(prediction - ee_ori_gt)
                acc_test_loss += loss.item()
                acc_test_accuracy += accuracy.to('cpu')

                print(f'- Test [{i+1}/{self.n_val_set}] -')
                print(f"Loss: {loss.item():.4f}")
                print(f"Ground truth end-point: [{ee_ori_gt.tolist()[0]:.4f}, {ee_ori_gt.tolist()[1]:.4f}, {ee_ori_gt.tolist()[2]:.4f}]")
                print(f"Prediction end-point: [{prediction.tolist()[0]:.4f}, {prediction.tolist()[1]:.4f}, {prediction.tolist()[2]:.4f}]")
                print(f'Accuracy end-point: [{accuracy[0]:.4f}, {accuracy[1]:.4f}, {accuracy[2]:.4f}]')

        mean_test_loss = acc_test_loss / self.n_val_set
        mean_test_acc = acc_test_accuracy / self.n_val_set
        print("- Overall Results -")
        print(f"Mean test loss: {mean_test_loss:.6f}")
        print(f"Mean test accuracy: [{mean_test_acc[0]:.4f}, {mean_test_acc[1]:.4f}, {mean_test_acc[2]:.4f}]")


if __name__ == "__main__":
    # Define hyperparameter
    epochs = 500
    neurons = 32
    learning_rate = 0.001
    l2_regularization = 0.0001

    """
    # Start weights and biases for tracking the training process
    wandb.login(key="...")
    wandb.init(
        # set the wandb project where this run will be logged
        project="spherical_joystick_fk_nn",

        # track hyperparameters and run metadata
        config={
            "epochs": epochs,
            "neurons": neurons,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization
        })
    """

    # Train forward kinematics model
    model_trainer = ForwardKinematicsNNTrainer(num_epochs=epochs, n_neurons=neurons, learning_rate=learning_rate, l2_regu=l2_regularization)
    model_trainer.train()
    model_trainer.test()
