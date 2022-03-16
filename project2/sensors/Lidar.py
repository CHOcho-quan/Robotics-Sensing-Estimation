# Lidar Class
import numpy as np

class Lidar:
    """
    Lidar sensor class
    
    """
    def __init__(self):
        """
        Lidar sensor class Initialization

        """
        self.fov_ = 190 # Degree
        self.sangle_ = -5 
        self.eangle_ = 185
        self.res_ = self.fov_ / 285
        self.max_range_ = 67

    def Detect(self, rays, lidar_scan_limits=(1.0, 55.0)):
        """
        Detect rays, filter useless rays
        Input:
        rays - input 285 rays as numpy array
        Outputs:
        result - 285 x 3 array which contains 285 end points
            and end points for the 285 rays (Lidar Frame) [x, y, 0]
        obs - bool array indicating if there's obstacle

        """
        result_rays = []
        obs = []
        rmin, rmax = lidar_scan_limits
        for i, ray in enumerate(rays):
            if ray == 0.0:
                ray = rmax
            if rmin <= ray <= rmax:
                cur_xy = np.zeros(3)
                cur_xy[0] = ray * np.cos(np.deg2rad(self.sangle_ + i * self.res_))
                cur_xy[1] = ray * np.sin(np.deg2rad(self.sangle_ + i * self.res_))
                result_rays.append(cur_xy)
                obs.append(ray != rmax)
        
        result_rays = np.array(result_rays)
        obs = np.array(obs)
        return result_rays, obs
