import math, itertools, cv2
import numpy as np
from OccupancyMap import OccupancyMap
from sensors.Lidar import Lidar
from sensors.Encoder import Encoder
from sensors.FOG import FOG
from sensors.Camera import Camera

class Robot:
    """
    A Robot SLAM class
    
    """
    def __init__(self, map_config=None, occupied_thres=np.log(9), \
        n_particles=100, predict_sigma=np.diag([1e-5, 1e-5, 1e-6]), Neff=4, corr=5):
        """
        Initialization function for the robot
        
        """
        # Map
        if map_config is None:
            self.omap_ = OccupancyMap()
        else:
            self.omap_ = OccupancyMap(map_config["xmin"], map_config["xmax"], \
                map_config["ymin"], map_config["ymax"], map_config["res"], \
                    map_config["trust"], map_config["logmin"], map_config["logmax"])
        self.occupied_thres_ = occupied_thres
        self.map_texture_ = np.zeros((self.omap_.sizey_, self.omap_.sizex_, 3), dtype=np.float64)
        
        # Robot State
        self.cur_timestamp_ = 0

        # Sensors
        self.lidar_ = Lidar()
        self.camera_ = Camera()
        self.encoder_ = Encoder()
        self.fog_ = FOG()

        # Parameters
        self.vTl_ = np.array([[0.00130201, 0.796097, 0.605167, 0.8349],
                              [0.999999, -0.000419027, -0.00160026, -0.0126869],
                              [-0.00102038, 0.605169, -0.796097, 1.76416],
                              [0, 0, 0, 1]])
        self.vTf_ = np.array([[1, 0, 0, -0.335],
                              [0, 1, 0, -0.035],
                              [0, 0, 1, 0.78],
                              [0, 0, 0, 1]])
        self.vTc_ = np.array([[-0.00680499, -0.0153215, 0.99985, 1.64239],
                              [-0.999977, 0.000334627, -0.00680066, 0.247401],
                              [-0.000230383, -0.999883, -0.0153234, 1.58411],
                              [0, 0, 0, 1]]) # vehicle to camera

        # Particle Filter Setting
        self.particles_ = np.zeros((n_particles, 3))
        self.weights_ = np.ones(n_particles) / n_particles
        self.n_particles_ = n_particles
        self.neff_ = Neff
        self.corr_ = corr
        self.sigma_predict_ = predict_sigma

    def Init(self, fog, enc, lidar_rays):
        """
        Initialization for the robot
        Input:
        fog - sensor FOG observation
        enc - sensor encoder observation
        lidar - sensor lidar observation
        
        """
        self.fog_.Init(fog[0])
        self.cur_timestamp_ = enc[0]
        self.encoder_.Init(enc[0], enc[1], enc[2])
        self.UpdateOccupancyMap(lidar_rays)

    @property
    def State(self):
        return np.sum(self.weights_.reshape(-1, 1) * self.particles_, axis=0)
    
    def GetOccupancyMap(self):
        logodds = self.omap_.lambda_
        return (logodds > self.occupied_thres_).astype(np.int32)
    
    def yaw(self, rad):
        return np.array([[np.cos(rad), -np.sin(rad), 0],
                         [np.sin(rad), np.cos(rad), 0],
                         [0, 0, 1]])
    
    def Predict(self, fog, enc):
        """
        Particle Filter Predict Function
        Input:
        fog - sensor FOG observation
        enc - sensor encoder observation
        
        """
        vl, vr = self.encoder_.Update(enc[0], enc[1], enc[2])
        _, _, wy = self.fog_.Update(fog[0], fog[1], fog[2], fog[3])
        tau = (enc[0] - self.cur_timestamp_) * 1e-9
        self.cur_timestamp_ = enc[0]

        # Differential Drive Model
        v = (vl + vr) / 2.0
        
        for i, p in enumerate(self.particles_):
            u = np.zeros(3)
            u[0] = v * tau * np.sinc(fog[3] / (2 * math.pi)) * np.cos(p[2] + fog[3] / 2)
            u[1] = v * tau * np.sinc(fog[3] / (2 * math.pi)) * np.sin(p[2] + fog[3] / 2)
            u[2] = fog[3]

            u = np.random.multivariate_normal(u, self.sigma_predict_, 1)
            self.particles_[i] += np.squeeze(u)
        return u

    def UpdateOccupancyMap(self, lidar_rays, lidar_scan_limit=(1.0, 55.0), test=False):
        """
        Update Occupancy Map
        Input:
        lidar_rays - 285 x 3 lidar rays indicating end points

        """
        lidar_rays, obs = self.lidar_.Detect(lidar_rays, lidar_scan_limit)
        lidar_rays = np.hstack([lidar_rays, np.ones((lidar_rays.shape[0], 1))]) # Homogeneous Coordinate
        cur_state = self.State
        wTv = np.vstack([np.hstack([self.yaw(cur_state[2]), \
                         np.array([[cur_state[0], cur_state[1], 0]]).T]), np.array([0, 0, 0, 1])])
        ray_w = wTv @ self.vTl_ @ lidar_rays.T
        ray_w = ray_w.T[:, :3]
        self.omap_.Update(ray_w, obs, test)

    def UpdateTextureMap(self, l_img, r_img):
        """
        Updating Texture Map by stereo
        Input:
        l_img - left camera original image
        r_img - right camera original image
        
        """
        # Get Depth and pixels
        depth = self.camera_.Update(l_img, r_img)
        depth[np.isinf(depth)] = -1
        h, w = depth.shape[:2]
        v = (depth * np.array(range(h)).reshape(h, -1)).reshape(-1, 1)
        thresh_ind = np.squeeze(v > 0)
        v = v[thresh_ind]
        u = (depth * np.array(range(w)).reshape(-1, w)).reshape(-1, 1)
        u = u[thresh_ind]
        depth = depth.reshape(-1, 1)
        depth = depth[thresh_ind]
        pixels = np.hstack([u, v, depth]) # Dimension: 3 x d

        # Get world frame points
        points_c = np.linalg.inv(self.camera_.Intrinsic) @ pixels.T
        points_c = np.vstack([points_c, np.ones((1, points_c.shape[1]))])
        cur_state = self.State
        wTv = np.vstack([np.hstack([self.yaw(cur_state[2]), \
                         np.array([[cur_state[0], cur_state[1], 0]]).T]), np.array([0, 0, 0, 1])])
        points_w = wTv @ self.vTc_ @ points_c
        points_w = points_w.T[:, :3]

        # Thresh and project to map texture
        l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
        left = l_img.reshape(-1, 3)
        left = left[thresh_ind, :]
        in_map = self.omap_.InOMap(points_w)
        points_w = points_w[in_map, :]
        left = left[in_map, :]

        ground_points_mask = np.logical_and(points_w[:, 2] < 1.0,
                                            points_w[:, 2] > -1.0)
        dark_color_mask = np.logical_and(
                np.logical_and(left[:, 0] > 20, left[:, 1] > 20), left[:, 2] > 20)
        mask = np.logical_and(dark_color_mask, ground_points_mask)
        points_w, left = points_w[mask, :], left[mask, :]
        left = left / 255.0

        indices = self.omap_.GetCoordinates(points_w[:, [0, 1]])
#         self.map_texture_[indices[:, 1], indices[:, 0], :] *= 0.5
        self.map_texture_[indices[:, 1],
                          indices[:, 0], :] = left[:, :]

    def Update(self, lidar_rays, lidar_scan_limit=(1.0, 55.0)):
        """
        Particle Filter Update Function
        Input:
        lidar_rays - 285 x 3 lidar rays indicating end points
        lidar_scan_limit - limit range of the useful lidar scan

        """
        # Homogeneous Coordinate
        lidar_rays, _ = self.lidar_.Detect(lidar_rays, lidar_scan_limit)
        lidar_rays = np.hstack([lidar_rays, np.ones((lidar_rays.shape[0], 1))])

        # Update particles
        correlation = np.zeros_like(self.weights_)
        map_binary = self.GetOccupancyMap()
        for i, p in enumerate(self.particles_):
            wTv = np.vstack([np.hstack([self.yaw(p[2]), \
                             np.array([[p[0], p[1], 0]]).T]), np.array([0, 0, 0, 1])])
            ray_w = wTv @ self.vTl_ @ lidar_rays.T

            # Filter useless lidar rays
            lidar_scan_w = ray_w[ray_w[:, 2] > 0.1, :]
            lidar_scan_w = lidar_scan_w[self.omap_.InOMap(lidar_scan_w), :]
            lidar_scan_ind = self.omap_.GetCoordinates(lidar_scan_w)

            # n x n Map Correlation
            bias = {}
            corr = np.zeros(self.corr_ ** 2)
            for j, (bx, by) in enumerate( \
                itertools.product(range(-(self.corr_ // 2), (self.corr_ + 1) // 2), \
                                  range(-(self.corr_ // 2), (self.corr_ + 1) // 2))):
                bias[j] = (bx, by)
                try:
                    corr[j] = np.sum(map_binary[lidar_scan_ind[:, 1] + by, lidar_scan_ind[:, 0] + bx])
                except:
                    corr[j] = -1
            
            # Best Particle Place
            correlation[i] = corr.max()

        # PF update weights
        log_weights = np.log(self.weights_) + correlation
        log_weights -= log_weights.max() + np.log(np.exp(log_weights - log_weights.max()).sum())
        self.weights = np.exp(log_weights)
        
        n_eff = 1.0 / np.sum(self.weights_ ** 2)
        if n_eff <= self.neff_:
            self.Resample()
    
    def Resample(self):
        """
        Particle Filter Resampling Function using SIR
        
        """
        # print("--- Resampling ---")
        indices = []
        W = np.cumsum(self.weights_)
        n = self.n_particles_
        self.weights = np.ones(n) / n
        new_particles = np.zeros((n, 3))

        # SIR Resample
        for i in range(n):
            p_ind = 0
            u = np.random.uniform()
            while u >= W[p_ind]:
                p_ind += 1
            new_particles[i] = self.particles_[p_ind]
        self.particles_ = new_particles

