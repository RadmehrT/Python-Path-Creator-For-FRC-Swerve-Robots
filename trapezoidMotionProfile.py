class trapezoidMotionProfile:
    
    def __init__(self, vi, vf, accel_t, noccel_t, deccel_t):
        self.vi
        self.vf
        self.accel_t
        self.noccel_t
        self.deccel_t
        

    def findAcceleration(self, vi, vf, t) -> float: # where t is time, this function is for both accel and deccel
        return (vf - vi) / t
        
    def getAccel_T(self) -> float: 
        return self.accel_t
    
    def getNoccel_T(self) -> float: 
        return self.noccel_t
    
    def getDeccel_t(self) -> float: 
        return self.deccel_t
    
    def getMaxAccel(self) -> float:
        return self.findAcceleration(vi, vf, self.accel_t)
    
    def getMaxDeccel(self) -> float:
        return self.findAcceleration(vf, vi, self.deccel_t) # vf is peak and vi is 0 so reversing them gives deccel period.
    
    def getNoccel(self):
        return 0 # no acceleration