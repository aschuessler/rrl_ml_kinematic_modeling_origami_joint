from inverse_kinematics_cross_entropy_method_fast import cem_based_optimization, loss_function
from forward_kinematics_nn import ForwardKinematicsNN
from dataset import ForwardKinematicsDataset

import torch
import numpy as np
import time
import pandas as pd


class Validation:
    def __init__(self, file_name):
        # Define forward kinematics neural network
        self.fk_nn_model_name = "models/231020_exp_4_spherical_joint_fk_ypr_best"
        self.fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
        self.fk_nn.load_state_dict(torch.load(self.fk_nn_model_name))
        self.fk_nn.eval()

        # Cross entropy method (CEM)
        self.n_samples = 1000
        self.num_opti_iters = 10
        self.num_elites = 100

        # Data set for validation
        self.dataset_name = "data/231020_experiment_4_sync.csv"
        self.dataset_val = ForwardKinematicsDataset(self.dataset_name, norm=False, train=False)

        # Save validation to file
        self.file_name = file_name

    def validate(self):
        with torch.no_grad():
            cumulative_loss_joint = 0.0
            cumulative_accuracy_joint = np.array([0.0, 0.0, 0.0, 0.0])
            cumulative_inference_time = 0.0
            for i, (joint_angles, ee_ori_gt) in enumerate(self.dataset_val):
                print(f"Evaluation of Sample {i+1}")
                t_prediction_start = time.time()
                pred_joint_angle, loss_ee, pred_orientation = cem_based_optimization(self.fk_nn, ee_ori_gt, self.n_samples, self.num_opti_iters, self.num_elites)
                t_prediction_end = time.time() - t_prediction_start

                loss_joint = np.sum(loss_function(pred_joint_angle, joint_angles.numpy(), len(self.dataset_val)))
                accuracy_joint = np.abs(pred_joint_angle - joint_angles.numpy())
                cumulative_loss_joint += loss_joint
                cumulative_accuracy_joint += accuracy_joint
                cumulative_inference_time += t_prediction_end

            mean_loss_joint = cumulative_loss_joint / len(self.dataset_val)
            mean_accuracy_joint = cumulative_accuracy_joint / len(self.dataset_val)
            mean_inference_time = cumulative_inference_time / len(self.dataset_val)

            print("--- Overall Results - IK prediction ---")
            print(f"Mean inference time: {mean_inference_time}")
            print(f"Mean loss joint prediction of IK: {mean_loss_joint:.6f}")
            print("Mean accuracy joint prediction:")
            print(f"[{mean_accuracy_joint[0]:.4f}, {mean_accuracy_joint[1]:.4f}, {mean_accuracy_joint[2]:.4f}, {mean_accuracy_joint[3]:.4f}]")

            # Save validation data to csv
            val_data = {"Mean inference time": pd.Series(mean_inference_time),
                        "Mean accuracy joint prediction": pd.Series(mean_accuracy_joint),
                        "Mean loss joint prediction": pd.Series(mean_loss_joint)}

            df_ik_val = pd.DataFrame(val_data)
            df_ik_val.to_csv(self.file_name)


if __name__ == "__main__":
    val = Validation("experiments/231107_ik_validation.csv")
    val.validate()
