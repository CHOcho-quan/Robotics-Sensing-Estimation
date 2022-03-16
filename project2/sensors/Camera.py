import cv2
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    """
    Camera Sensor Class

    """
    def __init__(self):
        """
        Initialization for Camera Sensor
        
        """
        self.width_ = 1280
        self.height_ = 560
        self.baseline_ = 475.143600050775 * 1e-3
        # Right Camera
        self.rcamera_matrix_ = \
            np.array([[8.1378205539589999e+02, 3.4880336220000002e-01, 6.1386419539320002e+02],
                      [0., 8.0852165574269998e+02, 2.4941049348650000e+02],
                      [0., 0., 1.]])
        self.rdistort_ = \
            np.array([-5.4921981799999998e-02, 1.4243657430000001e-01, 7.5412299999999996e-05,
                      -6.7560530000000001e-04, -8.5665408299999996e-02 ])
        self.rrectify_ = \
            np.array([[9.9998812489422739e-01, 2.4089155522231892e-03, -4.2364131513853301e-03],
                      [-2.4186483057924992e-03, 9.9999444433315865e-01, -2.2937835970734117e-03],
                      [4.2308640843048539e-03, 2.3040027516418276e-03, 9.9998839561287933e-01]])
        self.rprojection_ = \
            np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, -3.6841758740842312e+02],
                      [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        self.rintrinsic_ =  np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02], 
                                      [0., 7.7537235550066748e+02, 2.5718049049377441e+02],
                                      [0., 0., 1.]])

        # Left Camera
        self.lcamera_matrix_ = \
            np.array([[8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02],
                      [0., 8.1156803828490001e+02, 2.6347599764440002e+02],
                      [0., 0., 1.]])
        self.ldistort_ = \
            np.array([-5.6143027800000002e-02, 1.3952563200000001e-01, -1.2155906999999999e-03,
                      -9.7281389999999998e-04, -8.0878168799999997e-02 ])
        self.lrectify_ = \
            np.array([[9.9996942080938533e-01, 3.6208456669806118e-04, -7.8119357978017733e-03],
                      [-3.4412461339106772e-04, 9.9999729518344416e-01, 2.3002617343453663e-03],
                      [7.8127475572218850e-03, -2.2975031148170580e-03, 9.9996684067775188e-01]])
        self.lprojection_ = \
            np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.], 
                      [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.],
                      [0., 0., 1.,0.],
                      [0., 0., 0., 1.]])
        self.lintrinsic_ =  np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02], 
                                      [0., 7.7537235550066748e+02, 2.5718049049377441e+02],
                                      [0., 0., 1.]])

    @property
    def Intrinsic(self):
        return self.lintrinsic_

    def Update(self, l_img, r_img, test=False):
        """
        Estimate Depth Image by Stereo Camera
        Input:
        l_img - Left Camera Image 1280 x 560
        r_img - Right Camera Image 1280 x 560
        Output:
        depth - Depth Camera Image 1280 x 560
        
        """
        image_l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

        # Stereo Disparity Map
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
        disparity = stereo.compute(image_l_gray, image_r_gray)

        np.set_printoptions(suppress=True)
        depth = self.baseline_ * 7.7537235550066748e+02 / disparity
        if test:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(disparity, cmap='gray')
            ax2.imshow(depth, cmap='gray')
            plt.show()
        
        return depth

if __name__ == '__main__':
    l = cv2.imread("./data/image_left.png")
    r = cv2.imread("./data/image_right.png")
    cam = Camera()
    cam.Update(l, r, True)
