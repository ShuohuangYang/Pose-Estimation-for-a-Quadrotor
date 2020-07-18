# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import scipy as sp


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # constant
    # calculate em
    a_norm = np.linalg.norm(linear_acceleration)
    em = abs(a_norm/9.8 - 1)
    if em < 0.1:
        alpha = 1
    elif em > 0.2:
        alpha = 0
    else:
        alpha = -10 * em + 2

    # 1. integrating Gyro Measurements to get rotations
    # form skew symmetric matrix
    omega_hat = np.array([[0, -angular_velocity[2], angular_velocity[1]],
                          [angular_velocity[2], 0, -angular_velocity[0]],
                          [-angular_velocity[1], angular_velocity[0], 0]])
    I = np.identity(3)
    # curr_rotation = I + np.sin(dt)*omega_hat + (1-np.cos(dt))*omega_hat*omega_hat
    curr_rotation = sp.linalg.expm(omega_hat*dt)
    ini_Rot_mat = initial_rotation.as_matrix()
    Rotation_updated = np.dot(ini_Rot_mat, curr_rotation)

    # 2. calculate g'
    linear_acc = np.array(linear_acceleration)
    g_prime = np.dot(Rotation_updated, linear_acc)
    g_prime_norm = np.linalg.norm(g_prime)
    g_p_normalized = g_prime / g_prime_norm

    # 3. calculate q_acc
    # ex = np.array([1, 0, 0])
    # omega_acc = np.cross(g_p_normalized, ex)
    # costheta_acc = np.dot(g_p_normalized, ex) / (np.linalg.norm(g_p_normalized) * np.linalg.norm(ex))
    # theta = np.arccos(costheta_acc)
    # omega_acc_hat = np.array([[0, -omega_acc[2], omega_acc[1]],
    #                       [omega_acc[2], 0, -omega_acc[0]],
    #                       [-omega_acc[1], omega_acc[0], 0]])
    # Rotation_acc = I + np.sin(theta)*omega_acc_hat + (1-np.cos(theta))*omega_acc_hat*omega_acc_hat
    # Rotation_acc = Rotation.from_matrix(Rotation_acc)
    # q_acc = Rotation_acc.as_quat()

    q_acc = Rotation.from_quat([0, g_p_normalized[2] / np.sqrt(2 * (g_p_normalized[0] + 1)) , -g_p_normalized[1] / np.sqrt(2 * (g_p_normalized[0] + 1)), np.sqrt((g_p_normalized[0] + 1) / 2)])
    q_acc = q_acc.as_quat()

    # choose alpha and the mid way correction rotation
    qI = np.array([0,0,0,1])
    q_acc_p = (1-alpha)*qI + alpha*q_acc
    q_acc_p_norm = np.linalg.norm(q_acc_p)
    q_acc_p_normalized = q_acc_p / q_acc_p_norm
    q_acc_quat = Rotation.from_quat(q_acc_p_normalized)
    q_acc_matrix = q_acc_quat.as_matrix()
    # correct rotation
    Rotation_corrected = np.dot(q_acc_matrix, Rotation_updated)



    return Rotation.from_matrix(Rotation_corrected)



