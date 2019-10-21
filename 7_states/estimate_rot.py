#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import sys
sys.path.append('./code/')
from sigma_pts import Sigma_pts
import quat
import numpy as np
import scipy.io as sio
import params
from state import State
import matplotlib.pyplot as plt

def read_data_imu(num):
        return sio.loadmat("../data/imu/imuRaw" + str(num) + ".mat")


def read_data_vicon(num):
    return sio.loadmat("../data/vicon/viconRot" + str(num) + ".mat")

bias_g = [371.3300, 374.7600, 376.2000]
bias_a = [511,500, 500]
sens_a = 34.7455
sens_g = 300


def convert_measurements(acc, gyro):
    acc_corr = (acc - bias_a)*3300/1023/sens_a # Found from examining the data
    gyro_corr = (gyro - bias_g) * 3300 / 1023 /sens_g
    return np.hstack((acc_corr, gyro_corr))


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0, :] * R[0, 0, :] + R[1, 0, :] * R[1, 0, :])
    singular = sy < 1e-6
    x = np.arctan2(R[2, 1, :], R[2, 2, :])
    y = np.arctan2(-R[2, 0, :], sy)
    z = np.arctan2(R[1, 0, :], R[0, 0, :])

    return np.array([x, y, z])

def estimate_rot(data_num=1):
        # Read the data
        imu_data = read_data_imu(data_num)
        q = quat.Quaternion(np.array([1, 0, 0, 0]))
        omega = np.array([0, 0, 0])
        curr_state = State(q, omega)
        param = params.Params()
        ts = np.squeeze(imu_data['ts'])
        real_measurement = np.squeeze(imu_data['vals']).transpose()
        size_ts = ts.shape[0]
        sigma = Sigma_pts()
        rpy = []
        for i in range(1, size_ts):
            dt = ts[i] - ts[i-1]
            sigma.find_points(curr_state, dt)
            sigma.find_measurements()
            corrected_measurements = convert_measurements(real_measurement[i,:3], real_measurement[i,3:])
            curr_state.kalman_update(sigma, corrected_measurements)
            rpy.append(curr_state.q.quat2rpy())

        # plot vicon data
        vicon_data = read_data_vicon(data_num)
        v_ts = np.squeeze(vicon_data['ts'])
        x, y, z = rotationMatrixToEulerAngles(vicon_data['rots'])
        # plt.plot(v_ts, x, 'r')
        plt.figure()
        vicon, = plt.plot(v_ts, x, label="Vicon")
        ukf, = plt.plot(ts[1:], np.asarray(rpy)[:, 0], label="UKF")
        plt.legend(handles=[vicon, ukf])
        plt.title("Roll")
        plt.figure()
        vicon, =  plt.plot(v_ts, y, label="Vicon")
        ukf, =  plt.plot(ts[1:], np.asarray(rpy)[:, 1], label="UKF")
        plt.legend(handles=[vicon, ukf])
        plt.title("Pitch")
        plt.figure()
        vicon, = plt.plot(v_ts, z, label="Vicon")
        ukf, =  plt.plot(ts[1:], np.asarray(rpy)[:, 2], label="UKF")
        plt.legend(handles=[vicon, ukf])
        plt.title("Yaw")
        plt.show()


if __name__ == '__main__':
    estimate_rot(1)
