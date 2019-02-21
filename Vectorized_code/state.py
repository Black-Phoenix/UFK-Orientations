import copy
import quat
import numpy as np


def process_model(states, dt):
    q = quat.multiply(states[:,:4], quat.init_omega(states[:,4:], dt))
    w = states[:,4:]
    return np.column_stack((q, w))

def acc_model(states):
    # maybe should be a unit quat
    # maybe +ve g
    g = np.array([[0, 0, 0, 9.8]])
    inv_states = copy.deepcopy(states[:,:4])
    inv_states[:,1:4] = -inv_states[:, 1:4]
    g_body = quat.multiply(quat.multiply(inv_states, g), states[:, :4])
    return g_body[:,1:4]

def sensor_model(states):
    return np.column_stack((acc_model(states), states[:, 4:])).squeeze()


def kalman_update(sigma_pts, measurement):
    v = measurement - sigma_pts.Z_mean
    kv = np.matmul(sigma_pts.K, v)
    q = quat.multiply(sigma_pts.Y_qmean[np.newaxis, :], quat.init_omega(kv[np.newaxis, :3], 1))
    w = sigma_pts.Y_mean[4:] + kv[3:]
    return np.hstack((q.flatten(), w))


def old_acc_model(state):
    # maybe should be a unit quat
    # maybe +ve g
    g = np.array([0, 0, 0, 9.8])
    quat_inv = copy.deepcopy(state[:4])
    quat_inv[1:4] = -quat_inv[1:4]
    g_body = quat.multiply(quat.multiply(state[np.newaxis, :4], g[np.newaxis, :] ), quat_inv[np.newaxis, :])
    return g_body.flatten()[1:]

def old_gyro_model(self, noise):
    return self.w + noise

def old_sensor_model(state):
    return np.column_stack((old_acc_model(state), state[4:]))

