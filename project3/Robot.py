import numpy as np
import scipy
from scipy.linalg import expm

class Robot:
    """
    A Robot Class to Perform VIO

    """
    def __init__(self, cur_timestamp, n_landmarks, imu_T_cam, baseline,
                       cam_intrinsic, prior_sigma=np.zeros((6, 6)), motion_noise=1e-3 * np.eye(6),
                       observation_noise=100 * np.eye(4), prior_landmark_sigma=1e-2 * np.eye(3)):
        """
        Initialization for the robot class
        
        """
        self.cur_timestamp_ = cur_timestamp
        self.state_ = np.eye(4)

        # Landmark Settings
        self.n_landmarks_ = n_landmarks
        self.landmarks_ = np.zeros((n_landmarks, 3))
        self.initialized_mask_ = np.zeros((n_landmarks), dtype=bool)
        self.initialized_maxid_ = 0

        # Robot Settings
        # Need a 180 deg flip on x-axis
        self.iTo_ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ imu_T_cam
        self.K_ = cam_intrinsic
        self.baseline_ = baseline
        self.Ks_ = np.block([[self.K_[:2, :], np.array([[0, 0]]).T], \
                            [self.K_[:2, :], np.array([[-self.K_[0, 0] * self.baseline_, 0]]).T]])

        # Sigmas
        self.sigma_state_ = np.block([[prior_sigma, np.zeros((6, 3 * n_landmarks))],
                                      [np.zeros((3 * n_landmarks, 6)),
                                       np.kron(np.eye(n_landmarks), prior_landmark_sigma)]])
        self.W_ = motion_noise
        self.V_ = observation_noise

    @property
    def wTo(self):
        # Since state is wTi
        return self.state_ @ self.iTo_

    @property
    def oTw(self):
        return np.linalg.inv(self.wTo)

    def Project(self, q):
        """
        Projection function w.r.t. q
        Input:
        q - Input argument for the Projection function = oTiT^{-1}mj

        """
        return q / q[2, :]

    def dProjection(self, q):
        """
        Calculate derivative of the projection function w.r.t. q
        Input:
        q - Input argument for the Projection function = oTiT^{-1}mj
        Output:
        dP - Derivative of the projection function

        """
        q = q.reshape(-1)
        return (1 / q[2]) * np.array([[1, 0, -q[0] / q[2], 0], [0, 1, -q[1] / q[2], 0], \
                                      [0, 0, 0, 0], [0, 0, -q[3] / q[2], 1]])

    def Odot(self, x):
        """
        Returns a O-dot matrix of a vector x
        Input:
        x - 1 x 4 vector to get O-dot matrix
        Output:
        Ox - 4 x 6 O-dot matrix of x

        """
        return np.block([[np.eye(3), -self.SkewMatrix(x[:3])], [np.zeros((1, 6))]])

    def SkewMatrix(self, u):
        """
        Returns a skew-symmetric matrix of a vector u
        Input:
        u - 3 x 1 vector to get skew symmetric matrix
        Output:
        U_skew - 3 x 3 skew symmetric matrix of u

        """
        U_skew = np.array([[0, -u[2][0], u[1][0]],
                           [u[2][0], 0, -u[0][0]],
                           [-u[1][0], u[0][0], 0]])
        return U_skew

    def SuccMatrix(self, u):
        """
        Returns a succ matrix of a 6 x 1 vector u
        Input:
        u - 6 x 1 vector indicating linear and angular velocity
        Output:
        U_succ - 6 x 6 succ matrix of u

        """
        U_succ = np.zeros((6, 6))
        W_skew = self.SkewMatrix(u[3:])
        V_skew = self.SkewMatrix(u[:3])

        U_succ[:3, :3] = W_skew
        U_succ[3:6, 3:6] = W_skew
        U_succ[:3, 3:6] = V_skew
        return U_succ
        
    def TwistMatrix(self, u):
        """
        Returns a twist matrix of a 6 x 1 vector u
        Input:
        u - 6 x 1 vector indicating linear and angular velocity
        Output:
        U_hat - 4 x 4 twisted matrix of u
        
        """
        U_hat = np.zeros((4, 4))
        U_hat[:3, :3] = self.SkewMatrix(u[3:])
        U_hat[:3, 3] = np.squeeze(u[:3])
        return U_hat
    
    def Feat2World(self, feat):
        """
        Convert a 4 x n feature to world coordinate
        Input:
        feat - 4 x n vector of features, no undetected features here
        Output:
        feat_w - 4 x n vector of the world coordinate of landmarks

        """
        feat_w = np.ones((4, feat.shape[1]))
        feat_w[0, :] = (feat[0, :] - self.K_[0, 2]) * self.baseline_ / (feat[0, :] - feat[2, :]) # x coord
        feat_w[1, :] = (feat[1, :] - self.K_[1, 2]) \
            * self.K_[0, 0] * self.baseline_ / ((feat[0, :] - feat[2, :]) * self.K_[1, 1]) # y coord
        feat_w[2, :] = self.K_[0, 0] * self.baseline_ / (feat[0, :] - feat[2, :]) # z coord

        feat_w = self.wTo @ feat_w
        return feat_w
    
    def InitLandmarks(self, feat):
        """
        Initialize landmarks by given features
        Input:
        feat - 4 x n vector of features, [-1, -1, -1, -1] means not detecting

        """
        observed = np.array(np.where(feat.sum(axis=0) > -4.0), dtype=np.int32).reshape(-1)
        if observed.size == 0:
            return observed
        
        # There are some landmark observations
        not_initialized = observed[np.invert(self.initialized_mask_[observed])]
        if not_initialized.size == 0:
            return observed

        # There are some not initialized observations
        self.initialized_mask_[not_initialized] = True
        feat = feat[:, not_initialized]
        feat_w = self.Feat2World(feat)
        self.landmarks_[not_initialized, :] = feat_w[:3, :].T
        
        self.initialized_maxid_ = max(observed.max() + 1, self.initialized_maxid_)
        return observed
        
    def EKFPredict(self, u_t, cur_timestamp):
        """
        EKF Prediction function for VIO
        Input:
        u_t - current input 6 x 1 vector containing linear & angular velocity
        cur_timestamp - current timestamp
        
        """
        # Time duration
        tau  = cur_timestamp - self.cur_timestamp_
        self.cur_timestamp_ = cur_timestamp
        
        # Predict next Gaussian Distribution
        self.state_ = self.state_ @ expm(tau * self.TwistMatrix(u_t))
        expm_succ_ut = expm(-tau * self.SuccMatrix(u_t))
        self.sigma_state_[:6, :] = expm_succ_ut @ self.sigma_state_[:6, :]
        self.sigma_state_[:, :6] = self.sigma_state_[:, :6] @ expm_succ_ut.T
        self.sigma_state_[:6, :6] += self.W_

    def UpdateMapOnly(self, feat):
        """
        EKF Update function for Mapping Only
        Input:
        feat - 4 x n current observation matrix including n feature landmarks
        Note [-1, -1, -1, -1] for not detected

        """
        observed = self.InitLandmarks(feat)
        # Nothing to be updated
        if observed.size == 0:
            return True
        
        # Update landmarks
        # Observation Matrix H should be (4 * n_feat, 3 * n_landmarks + 6), 6 extra for state of robot
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        n_feat = observed.size
        n_landmarks = self.initialized_maxid_
        
        x_lm = np.hstack([self.landmarks_[observed, :], np.ones((n_feat, 1))])
        Ht = np.zeros((4 * n_feat, 3 * n_landmarks))

        for i in range(n_feat):
            feat_i = observed[i]

            Ht[i * 4:(i + 1) * 4, feat_i * 3:(feat_i + 1) * 3] = \
                self.Ks_ @ self.dProjection(self.oTw @ x_lm[i, :]) @ self.oTw @ P.T

        # Landmark Sigma
        x_update_lm = self.landmarks_[:n_landmarks, :]
        P_landmark = self.sigma_state_[6:6 + 3 * n_landmarks, 6:6 + 3 * n_landmarks]

        # Innovation Term
        predicted_obs = (self.Ks_ @ self.Project(self.oTw @ x_lm.T)).ravel(order='F').reshape(-1, 1)
        innovation = feat[:, observed].ravel(order='F').reshape(-1, 1) - predicted_obs

        # Kalman Gain
        V = np.kron(np.eye(n_feat), self.V_)
        Kt = P_landmark @ Ht.T @ np.linalg.inv(Ht @ P_landmark @ Ht.T + V)
        x_update_lm += (Kt @ innovation)[:, 0].reshape(-1, 3)
        P_landmark = (np.eye(Kt.shape[0]) - Kt @ Ht) @ P_landmark

        # Update to our state
        self.landmarks_[:n_landmarks, :] = x_update_lm
        self.sigma_state_[6:6 + 3 * n_landmarks, 6:6 + 3 * n_landmarks] = P_landmark

    def EKFUpdate(self, feat):
        """
        EKF Update function for VIO
        Input:
        feat - 4 x n current observation matrix including n feature landmarks
        Note [-1, -1, -1, -1] for not detected

        """
        observed = self.InitLandmarks(feat)
        # print(observed)
        # Nothing to be updated
        if observed.size == 0:
            return True

        # Update landmarks
        # Observation Matrix H should be (4 * n_feat, 3 * n_landmarks + 6), 6 extra for state of robot
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        n_feat = observed.size
        n_landmarks = self.initialized_maxid_

        x_lm = np.hstack([self.landmarks_[observed, :], np.ones((n_feat, 1))])
        Ht = np.zeros((4 * n_feat, 6 + 3 * n_landmarks))

        for i in range(n_feat):
            feat_i = observed[i]

            Ht[i * 4:(i + 1) * 4, 6 + feat_i * 3:6 + (feat_i + 1) * 3] = \
                self.Ks_ @ self.dProjection(self.oTw @ x_lm[i, :]) @ self.oTw @ P.T

            Ht[i * 4:(i + 1) * 4, :6] = \
                -self.Ks_ @ self.dProjection(self.oTw @ x_lm[i, :]) @ np.linalg.inv(self.iTo_) \
                    @ self.Odot((np.linalg.inv(self.state_) @ x_lm[i, :]).reshape(-1, 1))

        # Landmark Sigma
        x_update_lm = self.landmarks_[:n_landmarks, :]
        P_landmark = self.sigma_state_[:6 + 3 * n_landmarks, :6 + 3 * n_landmarks]

        # Innovation Term
        predicted_obs = (self.Ks_ @ self.Project(self.oTw @ x_lm.T)).ravel(order='F').reshape(-1, 1)
        innovation = feat[:, observed].ravel(order='F').reshape(-1, 1) - predicted_obs
        # print("PREOBS", predicted_obs.shape)

        # Kalman Gain
        V = np.kron(np.eye(n_feat), self.V_)
        # print("HtMax", np.max(Ht @ P_landmark @ Ht.T))

        Kt = P_landmark @ Ht.T @ np.linalg.inv(Ht @ P_landmark @ Ht.T + V)
        # print("Kt", Kt.shape)
        x_update_lm += (Kt @ innovation)[6:, 0].reshape(-1, 3)
        P_landmark = (np.eye(Kt.shape[0]) - Kt @ Ht) @ P_landmark

        # Update to our state
        # print("Update", (Kt @ innovation)[:6, 0])
        self.landmarks_[:n_landmarks, :] = x_update_lm
        innov_state = (Kt @ innovation)[:6, 0].reshape(-1, 1)
        self.state_ = self.state_ @ scipy.linalg.expm(self.TwistMatrix(innov_state))
        # print("State:", self.state_)
        self.sigma_state_[:6 + 3 * n_landmarks, :6 + 3 * n_landmarks] = P_landmark
