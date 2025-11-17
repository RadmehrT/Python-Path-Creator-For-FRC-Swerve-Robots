class trapezoidMotionProfile:
    
    def __init__(self, vi: float, vf: float, max_accel: float, max_deccel: float | None = None):
        self.vi = vi              # initial velocity
        self.vf = vf              # maximum (cruise) velocity
        self.max_accel = max_accel
        self.max_deccel = max_deccel if max_deccel is not None else max_accel

    def getMaxAccel(self) -> float:
        return self.max_accel
    
    def getMaxDeccel(self) -> float:
        return self.max_deccel
