import quat
import numpy as np


class State:
    def __init__(self, q, w):
        # assert q type is quat
        self.q = q
        self.w = w

    def process_model(self, dt, noise=None, update=False):
        """
        Noise is a Quaternion, state[0] is a Quaternion, state[1] is a np.array of size 1,3
        """
        if noise is not None:
            q = self.q * noise.q * quat.Quaternion.init_omega(self.w, dt)
            w = self.w + self.w
        else:
            q = self.q * quat.Quaternion.init_omega(self.w, dt)
            w = self.w
        if not update:
            return State(q, w)
        else:
            self.q = q
            self.w = w

    def copy(self):
        return State(self.q, self.w)

    def acc_model(self, noise):
        # maybe should be a unit quat
        # maybe +ve g
        g = quat.Quaternion([0, 0, 0, 9.8])
        g_body = (self.q * g * self.q.inv())
        return g_body.v + noise

    def gyro_model(self, noise):
        return self.w + noise

    def sensor_model(self):
        noise = np.zeros(6)
        return np.asarray([self.acc_model(noise[:3]), self.gyro_model(noise[3:])]).flatten()

    def tolist(self):
        return [*self.q.tolist(), *self.w]

    def kalman_update(self, sigma_pts, measurement):
        v = measurement - sigma_pts.Z_mean
        kv = np.matmul(sigma_pts.K, v)
        self.q = sigma_pts.Y_mean.q* quat.Quaternion.init_omega(kv[:3], 1)
        self.w = sigma_pts.Y_mean.w + kv[3:]
