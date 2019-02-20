import numpy as np


class Quaternion:
    def __init__(self, q):
        self.s = q[0]
        self.v = np.squeeze(np.array(q[1:]))

    @staticmethod
    def init_aa(axis, angle):
        return Quaternion([np.cos(angle / 2.0), axis * np.sin(angle / 2.0)])

    @staticmethod
    def init_omega(omega, dt=1):
        angle = np.linalg.norm(omega) * dt

        if np.linalg.norm(omega) == 0:
            axis = np.array([0, 0, 0])
        else:
            axis = np.divide(omega, np.linalg.norm(omega))
        return Quaternion.init_aa(axis, angle)

    def __mul__(self, other):
        return Quaternion(
            [self.s * other.s - self.v.dot(other.v), self.s * other.v + other.s * self.v + np.cross(self.v, other.v)])

    def quat2rpy(self):
        sinr_cosp = +2.0 * (self.s * self.v[0] + self.v[1] * self.v[2])
        cosr_cosp = +1.0 - 2.0 * (self.v[0] ** 2 + self.v[1] ** 2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = +2.0 * (self.s * self.v[1] - self.v[2] * self.v[0])
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # ???
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = +2.0 * (self.s * self.v[2] + self.v[0] * self.v[1])
        cosy_cosp = +1.0 - 2.0 * (self.v[1]**2 + self.v[2]**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    def inv(self):
        return Quaternion([self.s, -self.v])

    def toarray(self):
        return np.array([self.s, *self.v])

    def tolist(self):
        return [self.s, *self.v]

    def quat2vec(self):
        s = self.s/np.linalg.norm([self.s, *self.v])
        v = self.v / np.linalg.norm([self.s, *self.v])
        theta = 2 * np.arccos(s)
        return v*theta

