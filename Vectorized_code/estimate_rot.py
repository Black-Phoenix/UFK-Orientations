# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an extended kalman filter
import time

from sigma_pts import Sigma_pts
import quat
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import state


def read_data_imu(num):
    return sio.loadmat("./imu/imuRaw" + str(num) + ".mat")


def read_data_vicon(num):
    return sio.loadmat("./vicon/viconRot" + str(num) + ".mat")


# [373.739, 375.590, 377.0355]#
# [371.3300, 374.7600, 376.2000] Mine I guess

def params(sigma, n):
    global bias_g, bias_a, sens_a, sens_g
    if n in [1, 2, 3, 4]:
        bias_g = [371.3300, 374.7600, 400]
        bias_a = [500, 500, 500]
        sens_a = 330 / 9.8  # 34.7455
        sens_g = 2
        sigma.P = 0.00000000001 * np.eye(6)
        sigma.Q = np.eye(6) * 0.003
        sigma.Q[3:, 3:] = sigma.Q[3:, 3:] / 100000000
        sigma.R = np.eye(6) * 0.005
        sigma.R[3:, 3:] = sigma.R[3:, 3:]
    else:
        bias_g = [363, 355.7600, 310]
        bias_a = [501, 501, 500]
        sens_a = 303 / 9.8  # 34.7455
        sens_g = 2.8
        sigma.P = 0.0000000001 * np.eye(6)
        sigma.Q = np.eye(6) * 0.006 + np.ones((6,6))*0.0001
        sigma.Q[3:, 3:] = sigma.Q[3:, 3:] * 1000
        sigma.R = np.eye(6) * 0.005 + np.ones((6,6))*0.000001
        # sigma.R[3:, 3:] = sigma.R[3:, 3:]


def convert_measurements(acc, gyro):
    acc_corr = (acc - bias_a) * 3300 / 1023 / sens_a
    gyro_corr = (gyro - bias_g) * 3300 / 1023 * np.pi / 180 / sens_g
    return [-acc_corr[0], -acc_corr[1], acc_corr[2], gyro_corr[1], gyro_corr[2], gyro_corr[0]]
    # return np.hstack((acc_corr, gyro_corr))

    # return np.asarray([acc_corr, gyro_corr]).flatten()


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0, :] * R[0, 0, :] + R[1, 0, :] * R[1, 0, :])
    x = np.arctan2(R[2, 1, :], R[2, 2, :])
    y = np.arctan2(-R[2, 0, :], sy)
    z = np.arctan2(R[1, 0, :], R[0, 0, :])

    return np.array([x, y, z])


def estimate_rot(data_num=1):
    # Read the data
    imu_data = read_data_imu(data_num)
    curr_state = np.asarray([1, 0, 0, 0, 0, 0, 0])
    ts = np.squeeze(imu_data['ts'])
    real_measurement = np.squeeze(imu_data['vals']).transpose()
    size_ts = ts.shape[0]
    sigma = Sigma_pts()
    params(sigma, data_num)
    rpy = []
    for i in range(1, size_ts):
        # print(i)
        dt = ts[i] - ts[i - 1]
        sigma.find_points(curr_state, dt)
        sigma.find_measurements()
        corrected_measurements = convert_measurements(real_measurement[i, :3], real_measurement[i, 3:])
        curr_state = state.kalman_update(sigma, corrected_measurements)
        # inv_curr_state = np.array([curr_state, -curr_state[1:4]]).flatten()
        # rpy.append(quat.quat2rpy(curr_state[:4]))
        rpy.append(quat.rotmat2rpy(quat.quat2rotmat(curr_state[np.newaxis, :4])))
    # plot vicon data
    # if data_num in [6,5]:
    #     rpy = np.asarray(rpy)[-2600 - 700:-700, :]
    return np.asarray(rpy)[:, 0], np.asarray(rpy)[:, 1], np.asarray(rpy)[:, 2]
    # 2531


if __name__ == '__main__':
    data_num = 1
    t0 = time.time()
    r, p, y = estimate_rot(data_num)
    t1 = time.time()
    print(t1 - t0)
    imu_data = read_data_imu(data_num)
    curr_state = np.asarray([1, 0, 0, 0, 0, 0, 0])
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
    # plt.plot(v_ts, z, 'k')
    # for i in vicon_data
    plt.show()
