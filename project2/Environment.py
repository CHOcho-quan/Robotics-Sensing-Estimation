# Env Class
import os, cv2
import pickle, glob
import numpy as np
from pr2_utils import *

class Env:
    """
    Environment class for replaying sensor data
    
    """
    def __init__(self, path=None):
        """
        Initialization for Environment
        
        """
        if path is None:
            self.lidar_ = []
            self.cameral_ = []
            self.camerar_ = []
            self.fog_ = []
            self.encoder_ = []
        else:
            self.LoadData(path)

    def LoadData(self, pickle_file):
        """
        Load data directly from pickle
        Input:
        pickle_file - place of the stored pickle file
        
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        self.lidar_ = data["lidar"]
        self.cameral_ = data["camera_left"]
        self.camerar_ = data["camera_right"]
        self.fog_ = data["fog"]
        self.encoder_ = data["encoder"]

    def ReadData(self, folder):
        """
        Read data from CSV and images
        Input:
        folder - root folder of the data
        
        """
        # Lidar
        lidar_csv = pd.read_csv(folder + "/sensor_data/lidar.csv", header=None)
        lidar_data = lidar_csv.values[:, 1:]
        lidar_timestamp = lidar_csv.values[:, 0]
        print(f"Starting Timestamp (Lidar) {lidar_timestamp[0]:f}")
        for lidar_d in zip(lidar_timestamp, lidar_data):
            t, ld = lidar_d
            self.lidar_.append(np.hstack((t, ld)))

        # FOG
        fog_csv = pd.read_csv(folder + "/sensor_data/fog.csv", header=None)
        fog_data = fog_csv.values[:, 1:]
        fog_timestamp = fog_csv.values[:, 0]
        for fog_d in zip(fog_timestamp, fog_data):
            t, fd = fog_d
            self.fog_.append(np.hstack((t, fd)))

        # Encoder
        enc_csv = pd.read_csv(folder + "/sensor_data/encoder.csv", header=None)
        enc_data = enc_csv.values[:, 1:]
        enc_timestamp = enc_csv.values[:, 0]
        for enc_d in zip(enc_timestamp, enc_data):
            t, ed = enc_d
            self.encoder_.append(np.hstack((t, ed)))

        # Camera
        left_imgs = sorted(glob.glob(folder + "/stereo_images/stereo_left/*.png"))
        right_imgs = sorted(glob.glob(folder + "/stereo_images/stereo_right/*.png"))

        for path_l, path_r in zip(left_imgs, right_imgs):
            image_l = cv2.imread(path_l, 0)
            image_r = cv2.imread(path_r, 0)

            image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
            image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

            self.cameral_.append((int(os.path.basename(path_l).split('.')[0]), image_l))
            self.camerar_.append((int(os.path.basename(path_r).split('.')[0]), image_r))

            image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
            image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
            stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
            disparity = stereo.compute(image_l_gray, image_r_gray)

        # Store result with pickle
        with open("env.pickle", 'wb') as f:
            data = {
                "lidar" : self.lidar_,
                "fog" : self.fog_,
                "encoder" : self.encoder_,
                "camera_left" : self.cameral_,
                "camera_right" : self.camerar_
            }
            pickle.dump(data, f)

env = Env("env.pickle")
# env.ReadData("./data")