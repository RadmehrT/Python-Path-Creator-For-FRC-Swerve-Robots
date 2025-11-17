class Pose2D:
    
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading
        
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getHeading(self):
        return self.heading
