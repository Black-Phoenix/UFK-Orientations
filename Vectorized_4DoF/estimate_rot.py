import time
import matplotlib.pyplot as plt
import numpy as np
from quat import *
import scipy.io as sio


def read_data_imu(num):
    return sio.loadmat("./imu/imuRaw" + str(num) + ".mat")


def read_data_vicon(num):
    return sio.loadmat("./vicon/viconRot" + str(num) + ".mat")


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0, :] * R[0, 0, :] + R[1, 0, :] * R[1, 0, :])
    x = np.arctan2(R[2, 1, :], R[2, 2, :])
    y = np.arctan2(-R[2, 0, :], sy)
    z = np.arctan2(R[1, 0, :], R[0, 0, :])
    return x, y, z


def prediction(state, control, P, Q):
    qu = vec2quat(np.atleast_2d(control).T)

    S = np.linalg.cholesky(P+Q)
    n = P.shape[0]
    S = S*np.sqrt(0.5*n)
    W = np.array(np.hstack([S, -S]))

    noise_quaternions = vec2quat(W)
    X = multiply(noise_quaternions.T, state)

    Y = multiply(X, qu.T)

    next_state, error = q_mean(Y.T, state)

    next_cov = np.dot(error, error.T)/12.0

    return next_state, next_cov, Y, error


def convert_measurements(acc, gyro, data_num):
    sens_a = 330.0
    sf_a = 3300 / 1023.0 / sens_a
    # Make sure that the first point is [0,0,1]
    if data_num in [1, 2, 3, 4, 5]:
        bias_g = np.array([373.73, 375.6, 377])
        sens_g = 3.3
    else:
        bias_g = np.array([373.64, 375.6, 364.4])
        sens_g = 3.2
    acc_scale_factor = 3300 / 1023.0 / sens_a
    acc_bias = acc[0, :] - (np.array([0, 0, 1]) / acc_scale_factor)
    acc_corr = (acc - acc_bias) * acc_scale_factor
    gyro_scale_factor = 3300/1023/sens_g
    gyro_corr = (gyro - bias_g)*gyro_scale_factor*(np.pi/180)
    return acc_corr, gyro_corr
    #


def estimate_rot(data_num=1):
    # Data from dataset
    imu = read_data_imu(data_num)
    imu_np_data = np.array(imu['vals'], dtype=np.float64).T
    imu_ts = imu['ts'].T

    a_x = -np.array(imu_np_data[:, 0])
    a_y = -np.array(imu_np_data[:, 1])
    a_z = np.array(imu_np_data[:, 2])
    acc = np.array([a_x, a_y, a_z]).T

    g_x = np.array(imu_np_data[:, 4])
    g_y = np.array(imu_np_data[:, 5])
    g_z = np.array(imu_np_data[:, 3])
    gyro = np.array([g_x, g_y, g_z]).T
    acc_val, gyro_val = convert_measurements(acc, gyro, data_num)
    curr_state = np.array([1, 0, 0, 0])

    if data_num in [1, 2, 3, 4]:
        P = 0.00000000001 * np.identity(3)  # 0.00001
        Q = np.array([[0.003, 0, 0],
                      [0, 0.003, 0],
                      [0, 0, 0.003]])
        R = np.array([[0.005, 0, 0],
                      [0, 0.005, 0],
                      [0, 0, 0.005]])
    else:
        P = 0.000000001 * np.identity(3)  # 0.00001
        Q = np.array([[0.03, 0, 0],
                      [0, 0.03, 0],
                      [0, 0, 0.03]])
        R = np.array([[0.05, 0, 0],
                      [0, 0.05, 0],
                      [0, 0, 0.05]])
    rpy = []
    prev_timestep = imu_ts[0] - 0.01
    for i in range(1, imu_ts.shape[0]):
        control = gyro_val[i] * (imu_ts[i] - prev_timestep)
        prev_timestep = imu_ts[i]

        curr_state, P, sigma_points, error = prediction(curr_state, control, P, Q)

        Z_simga_pts = multiply(quat_inv(sigma_points), np.array([0, 0, 0, 1]))
        Z_simga_pts = multiply(Z_simga_pts, sigma_points)
        Z_simga_pts = Z_simga_pts[:, 1:]

        z_est = np.mean(Z_simga_pts, axis=0)

        Z_error = (Z_simga_pts - z_est).T
        P_zz = np.dot(Z_error, Z_error.T) / 12.0
        P_vv = P_zz + R
        P_xz = np.dot(error, Z_error.T) / 12.0
        K = np.dot(P_xz, np.linalg.inv(P_vv))
        v = np.transpose(acc_val[i] - z_est)
        Knu = vec2quat(np.atleast_2d(np.dot(K, v )).T)
        curr_state = multiply(Knu.T, curr_state).reshape(4, )
        P = P - np.dot(np.dot(K, P_vv), np.transpose(K))
        rpy.append(rotmat2rpy(quat2rotmat(curr_state)))

    rpy = np.array(rpy)
    # rpy[-650:, 2] = 0
    if data_num in [5, 6]:
        roll = np.array(rpy[-2530 - 650:-650, 0])
        pitch = np.array(rpy[-2530 - 650:-650, 1])
        yaw = np.array(rpy[-2530 - 650:-650, 2])
    else:
        roll = np.array(rpy[:, 0])
        pitch = np.array(rpy[:, 1])
        yaw = np.array(rpy[:, 2])

    return roll, pitch, yaw


if __name__ == '__main__':
    data_num = 1
    t0 = time.time()
    r, p, y = estimate_rot(data_num)
    t1 = time.time()
    print(t1 - t0)
    imu_data = read_data_imu(data_num)
    ts = np.squeeze(imu_data['ts'])
    vicon_data = read_data_vicon(data_num)
    v_ts = np.squeeze(vicon_data['ts'])
    x, a, z = rotationMatrixToEulerAngles(vicon_data['rots'])
    # plt.plot(v_ts, x, 'r')
    plt.figure()
    plt.plot(v_ts, x)
    plt.plot(ts[1:], r)
    plt.figure()
    plt.plot(v_ts, a)
    plt.plot(ts[1:], p)
    plt.figure()
    plt.plot(v_ts, z)
    plt.plot(ts[1:], y)
    plt.show()
