# Occupancy Map Class
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OccupancyMap:
    """
    Occupancy Map class
    NOTE: All the operations in this class is under world frame
    
    """
    def __init__(self, xmin=-100, xmax=100, ymin=-100, ymax=100, \
                 resolution=0.1, trust=0.8, logmin=-100, logmax=100):
        """
        Initialization for Occupancy Map

        """
        # Basic Parameters for Map
        self.xmin_ = xmin
        self.xmax_ = xmax
        self.ymin_ = ymin
        self.ymax_ = ymax
        self.res_ = resolution
        self.sizex_ = int(np.ceil((xmax - xmin) / self.res_ + 1))
        self.sizey_ = int(np.ceil((ymax - ymin) / self.res_ + 1))
        # Occupancy / Log-odds Map
        """
        Occupancy Map: 0 for free, 1 for obstacle, 2 for undetected
        """
        self.omap_ = np.zeros((self.sizey_, self.sizex_), dtype=np.int8) + 2
        self.lambda_ = np.zeros((self.sizey_, self.sizex_), dtype=np.float)
        self.logodds_min_ = logmin
        self.logodds_max_ = logmax
        self.logodd_free_ = np.log((1 - trust) / trust)
        self.logodd_obs_ = np.log(trust / (1 - trust))

    def InOMap(self, ind):
        """
        Test if grids are inside map
        Input:
        ind - input indices
        Output:
        in_map - array representing if grids are in map
        
        """
        return np.logical_and( \
            np.logical_and(self.xmin_ <= ind[:, 0], ind[:, 0] <= self.xmax_), \
            np.logical_and(self.ymin_ <= ind[:, 1], ind[:, 1] <= self.ymax_))

    def GetCoordinates(self, lidar_rays):
        """
        Get Coordinates from lidar
        Input:
        coord - 285 x 3 lidar rays
        Output:
        map_coord - 285 x 2 lidar map coordinates
        
        """
        lidar_scan_inds = np.hstack([
            np.ceil((lidar_rays[:, 0] - self.xmin_) / self.res_).reshape(-1, 1),
            np.ceil((lidar_rays[:, 1] - self.ymin_) / self.res_).reshape(-1, 1),
        ]).astype(np.int32)
        return lidar_scan_inds
        
    def Update(self, lidar_rays, obstacle, test=False):
        """
        Update according to sensor detection
        Input:
        lidar_rays - 285 x 3 Lidar rays in world frame
        obstacle - 285 x 1 indicating if there's obstacle

        """
        lidar_scan_inds = self.GetCoordinates(lidar_rays)

        # Get freespace & Update log-odds
        self.lambda_ += cv2.drawContours(image=np.zeros_like(self.lambda_),
                                         contours=[
                                             lidar_scan_inds.reshape((-1, 1, 2))
                                         ],
                                        contourIdx=-1,
                                        color=self.logodd_free_,
                                        thickness=-1)
        # Remove non-obstacle places
        lidar_obs = np.delete(lidar_scan_inds, np.argwhere(obstacle==0).reshape(-1, 1), axis=0)
        self.lambda_[lidar_obs[:, 1], lidar_obs[:, 0]] += \
            self.logodd_obs_ - self.logodd_free_
        self.lambda_ = np.clip(self.lambda_, self.logodds_min_, self.logodds_max_)
        if test:
            plt.imshow(self.lambda_)
            plt.show()
            plt.savefig("lambda.png", dpi=2000)
