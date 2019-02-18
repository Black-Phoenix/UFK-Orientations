import numpy as np


class Quaternion:
    def __init__(self, q):
        self.s = q[0]
        self.v = np.squeeze(np.array(q[1:]))

    @staticmethod
    def init_aa(axis, angle):
        return Quaternion([np.cos(angle/2.0), axis*np.sin(angle/2.0)])

    @staticmethod
    def init_omega(omega, dt):
        angle = np.linalg.norm(omega)*dt
        axis = np.divide(omega, np.linalg.norm(omega))
        if any(np.isnan(axis)):
            axis = np.array([0, 0, 0])
        return Quaternion.init_aa(axis, angle)

    def __mul__(self, other):
        return Quaternion([self.s*other.s- self.v.dot(other.v), self.s*other.v + other.s*self.v + np.cross(self.v, other.v)])

    def quat2rpy(self):
        pass
