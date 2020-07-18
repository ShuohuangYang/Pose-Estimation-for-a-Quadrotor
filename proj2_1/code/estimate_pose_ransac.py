# Imports

import numpy as np
from scipy.spatial.transform import Rotation


# %%

def estimate_pose(uvd1, uvd2, pose_iterations, ransac_iterations, ransac_threshold):
    """
    Estimate Pose by repeatedly calling ransac

    :param uvd1:
    :param uvd2:
    :param pose_iterations:
    :param ransac_iterations:
    :param ransac_threshold:
    :return: Rotation, R; Translation, T; inliers, array of n booleans
    """

    R = Rotation.identity()

    for i in range(0, pose_iterations):
        w, t, inliers = ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold)
        R = Rotation.from_rotvec(w.ravel()) * R

    return R, t, inliers


def ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold):
    # find total number of correspondences
    n = uvd1.shape[1]

    # initialize inliers all false
    best_inliers = np.zeros(n, dtype=bool)

    for i in range(0, ransac_iterations):
        # Select 3  correspondences
        selection = np.random.choice(n, 3, replace=False)

        # Solve for w and  t
        w, t = solve_w_t(uvd1[:, selection], uvd2[:, selection], R)

        # find inliers
        inliers = find_inliers(w, t, uvd1, uvd2, R, ransac_threshold)

        # Update best inliers
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers.copy()

    # Solve for w and t using best inliers
    w, t = solve_w_t(uvd1[:, best_inliers], uvd2[:, best_inliers], R)

    return w, t, find_inliers(w, t, uvd1, uvd2, R, ransac_threshold)



def find_inliers(w, t, uvd1, uvd2, R0, threshold):
    """

    find_inliers core routine used to detect which correspondences are inliers

    :param w: ndarray with 3 entries angular velocity vector in radians/sec
    :param t: ndarray with 3 entries, translation vector
    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2:  3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :param threshold: Threshold to use
    :return: ndarray with n boolean entries : Only True for correspondences that pass the test
    """

    n = uvd1.shape[1]
    I = np.identity(3)
    R0 = R0.as_matrix()
    delta_total = np.zeros(n)
    w = w.reshape((3,1))
    t = t.reshape((3,1))
    for i in range(n):
        omega_hat = np.array([[0, -w[2,:], w[1,:]],
                              [w[2,:], 0, -w[0,:]],
                              [-w[1,:], w[0,:], 0]])
        R = (I + omega_hat).dot(R0)

        # calculate Z_ratio: Z1/Z2
        p_first = np.array([uvd1[0, i], uvd1[1, i], 1]).reshape((3, 1))
        p_second = np.array([uvd2[0, i], uvd2[1, i], 1]).reshape((3,1))
        Z_a = np.array([0, 0, 1]).reshape((1, 3))
        Z_b = R.dot(p_second) + uvd2[2, i] * t
        Z_ratio = Z_a.dot(Z_b)  # Z1/Z2

        # calculate delta
        delta_a = np.array([[1, 0, 0],
                            [0, 1, 0]])
        delta_b = R.dot(p_second) + uvd2[2, i] * t
        delta_c = Z_ratio * p_first
        delta_curr = delta_a.dot(delta_b - delta_c)
        delta_curr_norm = np.linalg.norm(delta_curr)

        delta_total[i] = delta_curr_norm*100

    inlier = (delta_total < threshold)
    print(inlier.sum())
    print(len(inlier))


    return inlier

def solve_w_t(uvd1, uvd2, R0):
    """
    solve_w_t core routine used to compute best fit w and t given a set of stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    """
    R0 = R0.as_matrix()
    n = uvd1.shape[1]
    A_total = []
    b_total = []
    for i in range(n):
        y = R0.dot(np.array([uvd2[0,i], uvd2[1,i], 1]))
        b = -1 * np.array([[1, 0, -uvd1[0,i]],
                           [0, 1, -uvd1[1,i]]])
        b = b.dot(y)
        A_a = np.array([[1, 0, -uvd1[0,i]],
                      [0, 1, -uvd1[1,i]]])
        A_b = np.array([[0, y[2], -y[1], uvd2[2,i], 0, 0],
                           [-y[2], 0, y[0], 0, uvd2[2,i], 0],
                           [y[1], -y[0], 0, 0, 0, uvd2[2,i]]])
        A = A_a.dot(A_b)

        # convert A, b ndarray to list
        A = A.tolist()
        b = b.tolist()

        # append to the total A, b
        A_total = A_total+A
        b_total = b_total+b

    A_total = np.array(A_total)
    b_total = np.array(b_total)
    x = np.linalg.lstsq(A_total, b_total, rcond=None)[0]
    w = x[0:3].reshape((3, 1))
    t = x[3:].reshape((3, 1))


    return w, t
