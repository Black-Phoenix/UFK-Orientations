#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import quat
import numpy as np
import scipy.io as sio
import params
def read_data(num):
        return sio.loadmat("./vicon/viconRot" + str(num) + ".mat")


def process_model(state, noise, dt):
    """
    Noise is a Quaternion, state[0] is a Quaternion, state[1] is a np.array of size 1,3
    """
    state[0] = state[0] * noise[0] * quat.Quaternion.init_omega(state[1], dt)
    state[1] = state[1] + noise[1]


def  meaurement_model():
    pass

def estimate_rot(data_num=1):
        # Read the data
        imu_data = read_data(data_num)
        q = quat.Quaternion(np.array([1, 0, 0, 0]))
        omega = np.array([0, 0, 0])
        state = [q, omega]
        param = params.Params()
        ts = np.squeeze(imu_data['ts'])
        size_ts = ts.shape[0]
        for i in range(1, size_ts):
            dt = ts[i] - ts[i-1]
            omega_noise = np.random.normal(0, param.Q, 3)
            q_noise = quat.Quaternion.init_omega(np.random.normal(0, param.Q, 3), 1)
            noise = [q_noise, omega_noise]
            process_model(state, noise, dt)
            sensor_update(imu_data)

        return roll,pitch,yaw
estimate_rot(1)
