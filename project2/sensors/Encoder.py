# Encoder Class
import math
from time import time

class Encoder:
    """
    Encoder Sensor Class
    
    """
    def __init__(self):
        """
        Initialization for Encoder
        
        """
        self.res_ = 4096
        self.ld_ = 0.623479
        self.rd_ = 0.622806
        self.wheel_base_ = 1.52439
        self.cur_timestamp_ = 0
        self.cur_lc_ = 0
        self.cur_rc_ = 0

    def Init(self, timestamp, lc, rc):
        """
        Init before using encoder
        Input:
        timestamp - current timestamp
        lc - current left count
        rc - current right count
        
        """
        self.cur_timestamp_ = timestamp
        self.cur_lc_ = lc
        self.cur_rc_ = rc

    def Update(self, timestamp, lc, rc):
        """
        Update and get velocity of current time
        Input:
        timestamp - current timestamp
        lc - current left count
        rc - current right count
        Output:
        vt - current velocity
        
        """
        time_diff = (timestamp - self.cur_timestamp_) * 1e-9 # nano sec 2 sec
        vlt = math.pi * self.ld_ * (lc - self.cur_lc_) / (self.res_ * time_diff)
        vrt = math.pi * self.rd_ * (rc - self.cur_rc_) / (self.res_ * time_diff)
        self.cur_timestamp_ = timestamp
        self.cur_lc_ = lc
        self.cur_rc_ = rc

        return vlt, vrt
