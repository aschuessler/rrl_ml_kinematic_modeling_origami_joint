import numpy.random

from forward_kinematics_nn import ForwardKinematicsNN
import numpy as np
import torch
import time


def cem_based_optimization(fk_model, target_ori, num_samples, num_opti_iters, num_elites):
    """
    Optimization with CEM method. In each optimization iteration,
    the joint angle configurations are sampled from a normal distribution.
    The cost for each joint angle configuration is calculated using the loss function and the forward kinematics.
    """

    # Set the seed for reproducibility
    np.random.seed(10)

    # Sample joint angle configurations
    joint_angle_min = 60
    joint_angle_max = 120
    sampled_joint_angles = joint_angle_min + numpy.random.random((num_samples, 4)) * (joint_angle_max - joint_angle_min)

    joint_angle_dim = sampled_joint_angles.shape[-1]

    # Calculate the mean and standard deviation of the sampled action sequences
    joint_angle_mean, joint_angle_std_dev = sampled_joint_angles.mean(axis=0), sampled_joint_angles.std(axis=0)

    # Define for returning None value if nothing was computed
    cumulative_loss = None
    sampled_joint_angle_configurations = None
    cumulative_pred_oris = None

    # Optimization iterations
    for i in range(num_opti_iters):
        #print(f"Optimization Itearation [{i+1}/{num_opti_iters}]")

        # Sample from a Gaussian distribution using the means and standard deviations
        sampled_joint_angle_configurations = joint_angle_mean + joint_angle_std_dev * np.random.randn(num_samples, joint_angle_dim)
        joint_angle_configuration_norm = torch.tensor(((sampled_joint_angle_configurations - joint_angle_min) / (joint_angle_max - joint_angle_min)), dtype=torch.float32)

        # t_inference_loop_start = time.time()

        # Changed inference (without for loop)
        with torch.no_grad():
            pred_ori = fk_model(joint_angle_configuration_norm)

        # Compute the loss for each joint angle configuration
        cumulative_loss = loss_function(pred_ori.numpy(), target_ori.numpy(), num_samples)

        # Add the prediction and loss to the array
        cumulative_pred_oris = pred_ori

        # t_inference_loop_end = time.time() - t_inference_loop_start

        # Select the top k elites of the configuration samples with lowest cost
        top_k = np.argsort(cumulative_loss)[:num_elites]
        elite = sampled_joint_angle_configurations[top_k, :]

        # Recalculate the mean and variances using the elite (top k) action sequences
        joint_angle_mean, joint_angle_std_dev = elite.mean(axis=0), elite.std(axis=0)

    # Select the joint angle configuration with the lowest cost - greedy selection of elite with best cost
    arg_best_loss = np.argmin(cumulative_loss)
    best_loss = cumulative_loss[arg_best_loss]
    best_joint_configuration = sampled_joint_angle_configurations[arg_best_loss]
    best_prediction = cumulative_pred_oris[arg_best_loss]

    return best_joint_configuration, best_loss, best_prediction


def loss_function(ee_orientation, ee_orientation_target, n):
    loss = (1/n) * np.sum((ee_orientation - np.expand_dims(ee_orientation_target, axis=0)) ** 2, axis=1)    # Previously 1/2
    return loss


def main():
    # Define forward kinematics neural network
    fk_nn_model_name = "models/231020_exp_4_spherical_joint_fk_ypr_best"
    fk_nn = ForwardKinematicsNN(input_dim=4, output_dim=3, n_neurons=32)
    fk_nn.load_state_dict(torch.load(fk_nn_model_name))
    fk_nn.eval()

    # Cross entropy method (CEM)
    n_samples = 1000
    num_opti_iters = 10
    target_ori = torch.tensor([10.0, 0.0,  0.0])  # [yaw, pitch, roll]
    num_elites = 100

    t_start_prediction = time.time()
    best_joint_angle, best_loss, best_prediction = cem_based_optimization(fk_nn, target_ori, n_samples, num_opti_iters, num_elites)
    t_end_prediction = time.time() - t_start_prediction
    print("--- Cross Entropy Method Results ---")
    print(f"Joint angles: {best_joint_angle}")
    print(f"Loss: {best_loss}")
    print(f"Prediction: {best_prediction}")
    print(f"Target: {target_ori}")
    print(f"Prediction time: {t_end_prediction:.4f}")


if __name__ == "__main__":
    main()