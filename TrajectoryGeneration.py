import math
import numpy as np

class TrajectoryGeneration:
    
    
    def __init__(self, p0, p2, b): # where p0 is robot starting pose, p2 is ending pose, and b distance vertex of parabola is from midpoint of the line between p0 and p2
        
        self.p0
        self.p2
        self.b
        

    def calculateMidpoint (self) -> list[float]: # where index 0 is x and 1 is y. We will just denote this as p3
        
        x = (self.p2[0] + self.p0[0]) / 2 # basically just the normal (thing 1 + thing 2) / 2
        
        y = (self.p2[1] + self.p0[1]) / 2
        
        return [x, y]
    

    def calculate_p1(self, b) -> list[float]: # where index 0 is x and 1 is y. This is p1
        
        p3 = self.calculateMidpoint()
        
        x = b
        
        
        m = -((self.p2[0] - self.p0[0]) / (self.p2[1] - self.p0[1]))
        y = m(x - p3[0]) + p3[1]
        
        return [x, y]
    
    def find_cfn(self) -> float:
        
        p1 = self.calculate_p1(self.b)
        
        p0x = self.p0[0]
        p1x = p1[0]
        
        p0y = self.p0[1]
        p1y = p1[1]
        
        dx = self.p2[0] - self.p0[0]
        dy = self.p2[1] - self.p0[1]
        
        
        
        cfn_numerator = (-(p0y - p1y)(dx)) + ((p0x - p1x)(dy) ((math.pow(dy, 2)) + (math.pow(dx, 2)))) 
        
        cfn_denominator = (math.sqrt(math.pow(dy, 2) + math.pow(dx, 2)))*(math.pow((((-1((p0x - p1x))(dx)) - (p0y - p1y)(dy))), 2)) # check around line 65 in graph to figure out what this is
        
        return cfn_numerator / cfn_denominator
        
    
    def X_ParametricFunction(self, t) -> float:
        
        p1 = self.calculate_p1(self.b)
        
        dx = self.p2[0] - self.p0[0]
        dy = self.p2[1] - self.p0[1]
        
        L = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        
        cfn = self.find_cfn()
        
        X = ((((-dy * (cfn * math.pow(t, 2))) * L) + (t * L * dx))  /  (-(math.pow(dy, 2) + math.pow(dx, 2)))) + p1[0] # check line about 131
        
        return X
        
        
    def Y_ParametricFunction(self, t) -> float:
        
        p1 = self.calculate_p1(self.b)
        
        dx = self.p2[0] - self.p0[0]
        dy = self.p2[1] - self.p0[1]
        
        L = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        
        cfn = self.find_cfn()
        
        Y = (L ((-dy * t) + (-1 * (cfn * math.pow(t, 2)) * dx)) / (math.pow(dy, 2) + math.pow(dx, 2))) + p1[1]  # check line about 136
        
        return Y
    
    
    def calculate_p4(self, t) -> list[float]: # point that travels along the curve
        
        xt = self.X_ParametricFunction(t)
        yt = self.Y_ParametricFunction(t)
        
        
        return [xt, yt]    
    
    
    def bounds(self) -> list[float]: # where index of 0 is lower bound and 1 is upper bound
        
        p1 = self.calculate_p1(self.b)
        
        p0 = self.p0
        p2 = self.p2
        
        dx = self.p2[0] - self.p0[0]
        dy = self.p2[1] - self.p0[1]
        
        L = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        
        lowerBound = (((p0[0] - p1[0]) * dx) + ((p0[1] - p1[1]) * dy)) / L
        
        upperBound = (((p2[0] - p1[0]) * dx) + ((p2[1] - p1[1]) * dy)) / L
        
        return [lowerBound, upperBound]
    
    
    def integrandOfLengthAlongTheCurve(self, u) -> float:
        
        #instantaneous value
        p = 2 * (self.find_cfn()) # why p? Because I ran out of letters on my keyboard during the graphing process
        
        integrand = np.hypot(1, (p * u))
        
        return integrand
    
    def lengthAlongTheCurve(self, lowerBound, upperBound, steps=10000) -> float:
        step_size = (upperBound - lowerBound) / steps
        total_length = 0.0

        u = lowerBound

        for _ in range(steps):
            u_next = u + step_size

            f_u = self.integrandOfLengthAlongTheCurve(u)
            f_u_next = self.integrandOfLengthAlongTheCurve(u_next)

            # trapezoid area = average height * width
            slice_length = 0.5 * (f_u + f_u_next) * step_size

            total_length += slice_length
            u = u_next

        return total_length
        
        
        
    def lengthAlongTheCurveArray(self, lowerBound=bounds()[0], upperBound=bounds()[1], steps=10000) -> list[float]:
        step_size = (upperBound - lowerBound) / steps
        total_length = []

        u = lowerBound

        for _ in range(steps):
            u_next = u + step_size

            f_u = self.integrandOfLengthAlongTheCurve(u)
            f_u_next = self.integrandOfLengthAlongTheCurve(u_next)

            # trapezoid area = average height * width
            slice_length = 0.5 * (f_u + f_u_next) * step_size

            total_length.append(slice_length)
            u = u_next

        return total_length
    
    
    
    
    def time_Integrand(self, vi, deltaX, a) -> float: # solving for time. Yes I know that integrand is not the true term, but I'm tired
        
        discriminant = math.pow(vi, 2) + (2 * a * deltaX)
        
        if discriminant < 0:
            return None #not a physically possible step
        
        sqrt_disc = math.sqrt(discriminant)
        
        t_plus = (-vi + sqrt_disc) / a
        t_minus = (-vi - sqrt_disc) / a
        
        potential_times = [t for t in (t_plus, t_minus) if t > 0]
        
        if not potential_times:
            raise ValueError("no positive time root found")
        
        t = min(potential_times)
        
        return t
        

        