from Pose2D import Pose2D
from PathCreator import pathCreator
from trapezoidMotionProfile import trapezoidMotionProfile

P0 = Pose2D(7.56, 6.57, 3.13)
P2 = Pose2D(5.61, 5.94, -2.0)

mp = trapezoidMotionProfile(0.0, 2.0, 1.0, 0.0, 1.0)
pc = pathCreator([], mp)

for b in [5.05, 3.0, 1.0, 0.7, 0.6139621056823753]:
    broken, total_time = pc._simulate_time_for_curve([P0.getX(), P0.getY()], [P2.getX(), P2.getY()], b, steps=1000)
    print("b=", b, "broken=", broken, "total_time=", total_time)

