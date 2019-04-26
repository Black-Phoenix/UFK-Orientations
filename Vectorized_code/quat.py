import numpy as np


def init_aa(axis, angle):
    return np.hstack((np.cos(angle / 2.0)[:,np.newaxis], axis * np.repeat(np.sin(angle / 2.0)[:,np.newaxis], 3, axis=1)))


def init_omega(omega, dt=1):
    angle = np.linalg.norm(omega, axis=1) * dt
    axis = omega / np.repeat(np.linalg.norm(omega, axis=1)[:, np.newaxis], 3, axis=1)

    axis[np.isnan(axis[:,0]*axis[:,1]*axis[:,2])] = [0,0,0]
    return init_aa(axis, angle)


def multiply(q1, q2):
    if q1.shape[0] == 1:
        q_1 = np.repeat(q1[:, :, np.newaxis], q2.shape[0], axis=0).squeeze(axis=2)
    else:
        q_1 = q1
    if q2.shape[0] == 1:
        q_2 = np.repeat(q2[:, :, np.newaxis], q1.shape[0], axis=0).squeeze(axis=2)
    else:
        q_2 = q2
    return np.column_stack((q_1[:,0]*q_2[:,0] - q_1[:,1]*q_2[:,1] - q_1[:,2]*q_2[:,2] - q_1[:,3]*q_2[:,3],
                            q_1[:,0]*q_2[:,1] + q_1[:,1]*q_2[:,0] + q_1[:,2]*q_2[:,3] - q_1[:,3]*q_2[:,2],
                            q_1[:,0]*q_2[:,2] - q_1[:,1]*q_2[:,3] + q_1[:,2]*q_2[:,0] + q_1[:,3]*q_2[:,1],
                            q_1[:,0]*q_2[:,3] + q_1[:,1]*q_2[:,2] - q_1[:,2]*q_2[:,1] + q_1[:,3]*q_2[:,0],))


def rotmat2rpy(r):
    norm = np.sqrt(r[2,1,:]**2 + r[2,2,:]**2)
    roll_x = np.arctan(r[2,1,:]/r[2,2,:])
    pitch_y = np.arctan(-r[2,0,:]/norm)
    yaw_z = np.arctan(r[1,0,:]/r[0,0,:])
    return [roll_x, pitch_y, yaw_z]

def quat2rpy(q):
    sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = +1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = +2.0 * (q[0] * q[2] - q[3] * q[1])
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # ???
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = +1.0 - 2.0 * (q[2]**2 + q[3]**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]
def rotmat2rpy(r):
    norm = np.sqrt(r[2,1]**2 + r[2,2]**2)
    roll_x = np.arctan(r[2,1]/r[2,2])
    pitch_y = np.arctan(-r[2,0]/norm)
    yaw_z = np.arctan(r[1,0]/r[0,0])
    return [roll_x, pitch_y, yaw_z]
def quat2rotmat(q):
    qhat = np.zeros([3,3])
    qhat[0,1] = -q[:,3]
    qhat[0,2] = q[:,2]
    qhat[1,2] = -q[:,1]
    qhat[1,0] = q[:,3]
    qhat[2,0] = -q[:,2]
    qhat[2,1] = q[:,1]
    R = np.eye(3) + 2*np.dot(qhat, qhat) + 2*np.array(q[:,0])*qhat
    return R
def quat2vec(q):
    q = q/np.repeat(np.linalg.norm(q, axis=1)[:, np.newaxis], 4, axis=1)
    theta = 2 * np.arccos(q[:, 0])
    return q[:, 1:] * np.repeat(theta[:,np.newaxis], 3, axis=1)
