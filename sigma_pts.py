import numpy as np
import quat
from state import State


class Sigma_pts:
    def __init__(self):
        self.n = 6
        # Maybe change this to start with current state * eye
        self.Y = []
        self.X = []
        self.Z = []
        self.Y_raw = []
        self.Y_qmean = quat.Quaternion([1, 0, 0, 0])
        self.e = quat.Quaternion([1, 0, 0, 0])
        self.P_v = []
        self.P_vv = []
        self.P = np.eye(6)
        self.R = np.eye(6)*0.00001
        self.Q = np.eye(6)*0.001
        self.Y_cov = np.zeros((6, 6))
        self.cross_cov = np.zeros((6, 6))
        self.ev_i =[]
        self.Z_cov = np.zeros((6, 6))

    def init_ts(self):
        self.X = []
        self.Y = []
        self.Z = []
        self.Y_raw = []
        self.Y_cov = np.zeros((6, 6))
        self.cross_cov = np.zeros((6, 6))
        self.ev_i = []
        self.Z_cov = np.zeros((6, 6))

    def find_points(self, state, dt):
        # Clean the old points
        self.init_ts()
        # todo add noise (35)
        W = np.vstack((np.sqrt(2*self.n )*np.linalg.cholesky((self.P + self.Q)).transpose(),
                       -np.sqrt(2*self.n)*(np.linalg.cholesky((self.P + self.Q)).transpose())))
        for i in W:
            q_new = state.q * quat.Quaternion.init_omega(i[0:3], 1)
            w_new = state.w + i[3:]
            self.X.append(State(q_new, w_new))
            self.Y.append(self.X[-1].process_model(dt, None, False))
            self.Z.append(self.X[-1].sensor_model())
            self.Y_raw.append(np.asarray((*self.Y[-1].q.tolist(), *self.Y[-1].w)))
        self.Y_raw = np.asarray(self.Y_raw)
        # Normalize the error vector

    def find_mean(self):
        self.Y_mean = State(self.Y_qmean, np.mean(self.Y_raw[:, 4:], axis=0))
        self.Z_mean = np.mean(self.Z, axis=0)

    def quaternion_mean(self):
        iter = 0
        while (1):
            iter += 1
            self.ev_i = []
            for i in self.Y:
                curr_error = (i.q * self.Y_qmean.inv())
                self.ev_i.append(curr_error.quat2vec())
            self.e = quat.Quaternion.init_omega(np.mean(self.ev_i, axis=0))
            self.Y_qmean = self.e*self.Y_qmean
            if np.linalg.norm(np.mean(self.ev_i, axis=0)) < 0.01 or iter > 100:
                break


    def find_cov(self):
        # Maybe wrong
        r_w = np.asarray(self.ev_i)
        w_w = self.Y_raw[:, 4:] - self.Y_mean.w
        W = np.vstack((r_w.transpose(), w_w.transpose()))

        self.Y_cov = 1 / (2 * self.n) * np.matmul(W, W.transpose())
        # Z time!!!
        Z_new = (self.Z - self.Z_mean).transpose()
        self.Z_cov = 1 / (2 * self.n) * np.matmul(Z_new, Z_new.transpose())
        self.P_vv = self.Z_cov + self.R

        self.cross_cov = 1 / (2 * self.n) * np.matmul(W, Z_new.transpose())

    def find_measurements(self):
        self.quaternion_mean()
        self.find_mean()
        self.find_cov()

        self.K = np.matmul(self.cross_cov, np.linalg.inv(self.P_vv))
        self.P = self.Y_cov - np.matmul(np.matmul(self.K, self.P_vv), self.K.transpose())
