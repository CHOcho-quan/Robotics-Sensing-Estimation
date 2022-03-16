# FOG Class
class FOG:
    """
    Gyro Sensor Class
    
    """
    def __init__(self):
        """
        Initialization for Gyro Sensor Class

        """
        self.cur_timestamp_ = 0

    def Init(self, timestamp):
        """
        Init before using FOG sensor
        Input:
        timestamp - current timestamp
        dr - delta roll angle
        dp - delta pitch angle
        dy - delta yaw angle
        
        """
        self.cur_timestamp_ = timestamp

    def Update(self, timestamp, dr, dp, dy):
        """
        Update for Gyro Sensor
        Input:
        timestamp - current timestamp
        dr - delta roll angle
        dp - delta pitch angle
        dy - delta yaw angle
        Output:
        wr - angular velocity of roll
        wp - angular velocity of pitch
        wy - angular velocity of yaw
        
        """
        time_diff = (timestamp - self.cur_timestamp_) * 1e-9
        wr = dr / time_diff
        wp = dp / time_diff
        wy = dy / time_diff

        return wr, wp, wy
