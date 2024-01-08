import torch
import torch.nn as nn
from forward_kinematics_nn import ForwardKinematicsNN
from dataset import ForwardKinematicsDataset
import pandas as pd
import time


class Validation:
    def __init__(self, file_name):
        # Define forward kinematics neural network
        self.fk_nn_model_name = "models/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(f'{self.fk_nn_model_name}'))
        self.fk_nn.eval()

        # Validation dataset
        self.dataset_name = "data/231020_experiment_4_sync.csv"
        self.dataset_val_nn = ForwardKinematicsDataset(self.dataset_name, norm=True, train=False)

        # Loss function
        self.mse_loss_function = nn.MSELoss()

        # Save data to file
        self.file_name = file_name

    def validate(self):
        with torch.no_grad():
            acc_test_loss_nn = 0.0
            acc_test_accuracy_nn = torch.tensor([0.0, 0.0, 0.0])
            acc_inference_time = 0.0
            for i, (joint_angles, ee_ori_gt) in enumerate(self.dataset_val_nn):
                t_prediction_start = time.time()
                nn_pred = self.fk_nn(joint_angles)
                t_prediction_end = time.time() - t_prediction_start

                loss = self.mse_loss_function(nn_pred, ee_ori_gt)
                accuracy = torch.abs(nn_pred - ee_ori_gt)
                acc_test_loss_nn += loss.item()
                acc_test_accuracy_nn += accuracy.to('cpu')
                acc_inference_time += t_prediction_end

            mean_test_loss_nn = acc_test_loss_nn / len(self.dataset_val_nn)
            mean_test_accuracy_nn = acc_test_accuracy_nn / len(self.dataset_val_nn)
            mean_inference_time = acc_inference_time / len(self.dataset_val_nn)

            print("- Overall Results -")
            print(f"Mean inference time: {mean_inference_time}")
            print(f"Mean validation loss: {mean_test_loss_nn:.6f}")
            print(f"Mean validation accuracy: [{mean_test_accuracy_nn[0]:.4f}, {mean_test_accuracy_nn[1]:.4f}, {mean_test_accuracy_nn[2]:.4f}]")

            # Save validation data to csv
            val_data = {"Mean inference time": pd.Series(mean_inference_time),
                        "Mean accuracy ypr prediction": pd.Series(mean_test_accuracy_nn),
                        "Mean prediction loss": pd.Series(mean_test_loss_nn)}

            df_ik_val = pd.DataFrame(val_data)
            df_ik_val.to_csv(self.file_name)


if __name__ == "__main__":
    val = Validation("experiments/231024_fk_validation.csv")
    val.validate()
