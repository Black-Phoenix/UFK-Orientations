import numpy as np
import quat
from state import *


class Sigma_pts:
    def __init__(self):
        self.n = 6
        # Maybe change this to start with current state * eye
        self.Y = np.zeros((7, 12))
        self.X = np.zeros((7, 12))
        self.Z = np.zeros((6, 12))
        self.Y_qmean = np.asarray([1, 0, 0, 0])
        self.e_quat = [1, 0, 0, 0]
        self.P_v = []
        self.P_vv = []
        # # Debug
        # self.P = np.eye(6)
        # self.Q = np.eye(6)
        # self.R = np.eye(6)
        # prefered last
        # self.P = 0.00001*np.eye(6)
        # self.Q = np.array([[1, 0, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1]])
        # self.R = np.asarray([[ 1.33934153e+01, -7.36833777e-02,  8.25092263e-01,
        #    -4.09273277e-03,  3.52014573e-04, -2.70162289e-03],
        #   [-7.36833777e-02,  1.06409008e+01,  1.33043765e+00,
        #    -8.36878084e-03, -6.87278634e-04,  8.29904982e-03],
        #   [ 8.25092263e-01,  1.33043765e+00,  1.14240991e+01,
        #     1.47024750e-02,  8.18841831e-03,  1.99298735e-02],
        #   [-4.09273277e-03, -8.36878084e-03,  1.47024750e-02,
        #     2.97851997e-03,  5.42550260e-03,  1.05243245e-03],
        #   [ 3.52014573e-04, -6.87278634e-04,  8.18841831e-03,
        #     5.42550260e-03,  1.20122430e-01,  4.79788346e-03],
        #   [-2.70162289e-03,  8.29904982e-03,  1.99298735e-02,
        #     1.05243245e-03,  4.79788346e-03,  1.43097413e-01]])
        # self.R[3:,3:] = self.R[3:,3:]/100

        # Nandas
        self.P = 0.00000000001*np.eye(6)
        self.Q = np.eye(6) * 0.003
        self.Q[3:,3:] = self.Q[3:,3:]/100000000
        self.R = np.eye(6) * 0.005
        self.R[3:,3:] = self.R[3:,3:]
        # Mine
        # self.P = np.eye(6)
        # self.R = np.eye(6)*0.00001
        # self.R[3,3] /= 100
        # self.R[4, 4] /= 100
        # self.R[5, 5] /= 100
        # self.Q = np.eye(6)*0.001
        self.Y_cov = np.zeros((6, 6))
        self.cross_cov = np.zeros((6, 6))
        self.ev_i =[]
        self.Z_cov = np.zeros((6, 6))

    def init_ts(self):
        self.X = np.zeros((7,12))
        self.Y = []
        self.Z = []
        self.Y_cov = np.zeros((6, 6))
        self.cross_cov = np.zeros((6, 6))
        self.ev_i = []
        self.Z_cov = np.zeros((6, 6))

    def find_points(self, state, dt):
        # Clean the old points
        self.init_ts()
        W = np.vstack((np.sqrt(2*self.n )*np.linalg.cholesky((self.P + self.Q)).transpose(),
                       -np.sqrt(2*self.n)*(np.linalg.cholesky((self.P + self.Q)).transpose())))
        self.X[:4,:] = quat.multiply(state[np.newaxis, :4], quat.init_omega(W[:,:3])).transpose()
        self.X[4:, :] = (state[4:] + W[:,3:]).T
        self.X = self.X.T
        self.Y = process_model(self.X, dt)
        # for i in self.X:
        #     self.Z.append(old_sensor_model(i))
        self.Z = sensor_model(self.X)
        # Normalize the error vector

    def find_mean(self):
        self.Y_mean = np.hstack((self.Y_qmean.flatten(), np.mean(self.Y[:, 4:], axis=0)))
        self.Z_mean = np.mean(self.Z, axis=0)

    def quaternion_mean(self):
        iter = 0
        while (1):
            iter += 1
            Y_qmean_inv = np.hstack((self.Y_qmean[0], -self.Y_qmean[1:]))
            self.ev_i = quat.quat2vec(quat.multiply(self.Y[:,:4], Y_qmean_inv[np.newaxis, :]))
            self.e_quat = quat.init_omega(np.mean(self.ev_i, axis=0)[np.newaxis, :])
            self.Y_qmean = quat.multiply(self.e_quat, self.Y_qmean[np.newaxis, :]).squeeze()
            if np.linalg.norm(np.mean(self.ev_i, axis=0)) < 0.001 or iter > 10000:
                break
        self.Y_qmean = self.Y_qmean/np.linalg.norm(self.Y_qmean)

    def find_cov(self):
        r_w = self.ev_i
        w_w = self.Y[:, 4:] - self.Y_mean[4:]
        W = np.vstack((r_w.T, w_w.T))

        self.Y_cov = np.matmul(W, W.T)/ (2 * self.n)
        # Z time!!!
        Z_new = (self.Z - self.Z_mean).T
        self.Z_cov = np.matmul(Z_new, Z_new.T)/ (2 * self.n)
        self.P_vv = self.Z_cov + self.R

        self.cross_cov = np.matmul(W, Z_new.transpose())/ (2 * self.n)

    def find_measurements(self):
        self.quaternion_mean()
        self.find_mean()
        self.find_cov()

        self.K = np.matmul(self.cross_cov, np.linalg.inv(self.P_vv))
        self.P = self.Y_cov - np.matmul(np.matmul(self.K, self.P_vv), self.K.transpose())
