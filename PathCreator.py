from TrajectoryGeneration import TrajectoryGeneration
from trapezoidMotionProfile import trapezoidMotionProfile
from Pose2D import Pose2D

import numpy as np

class pathCreator: 
    
    def __init__(self, waypoints: list[Pose2D], motionProfile: trapezoidMotionProfile):
        self.waypoints = waypoints
        self.motionProfile = motionProfile
        
    def formatWaypoints(self, curve_intensities: list[float]) -> list[list[Pose2D, Pose2D, float]]:
        waypoints = self.waypoints
        
        formattedWaypoints = []
        
        for i in range(len(waypoints)):
            if i % 2 != 0:
                continue

            if i == 0:
                formattedWaypoints.append([[waypoints[i], waypoints[i + 1]], curve_intensities[0]])
            else:
                formattedWaypoints.append([[waypoints[i], waypoints[i + 1]], curve_intensities[i // 2]])
        return formattedWaypoints

    def findT(self, P0: list[float], P2: list[float], curveIntensity: float) -> float | None:
        trajectorygenerator = TrajectoryGeneration(P0, P2, curveIntensity)
        try:
            lower, upper = trajectorygenerator.bounds()
        except ZeroDivisionError:
            return None

        if not np.isfinite(upper):
            return None

        return upper
            
    def GenerateActualPath(self, P0: Pose2D, P2: Pose2D, curveIntensity: float, steps=100) -> list[Pose2D]:
        p0_vec = [P0.getX(), P0.getY()]
        p2_vec = [P2.getX(), P2.getY()]

        trajectorygenerator = TrajectoryGeneration(p0_vec, p2_vec, curveIntensity)

        try:
            lower_t, upper_t = trajectorygenerator.bounds()
        except ZeroDivisionError:
            return []

        if not (np.isfinite(lower_t) and np.isfinite(upper_t)) or upper_t == lower_t:
            return []

        poses: list[Pose2D] = []

        if steps <= 1:
            t_values = [lower_t]
        else:
            dt = (upper_t - lower_t) / (steps - 1)
            t_values = [lower_t + i * dt for i in range(steps)]

        start_heading = P0.getHeading()
        end_heading = P2.getHeading()

        for idx, t in enumerate(t_values):
            x = trajectorygenerator.X_ParametricFunction(t)
            y = trajectorygenerator.Y_ParametricFunction(t)

            if steps <= 1:
                progress = 0.0
            else:
                progress = idx / (steps - 1)

            if progress <= 0.75:
                ratio = progress / 0.75 if 0.75 > 0 else 0.0
                heading = start_heading + ratio * (end_heading - start_heading)
            else:
                heading = end_heading

            poses.append(Pose2D(x, y, heading))

        return poses

    def _simulate_time_for_curve(self, p0_vec: list[float], p2_vec: list[float], curveIntensity: float, steps=1000) -> tuple[bool, float]:
        tg = TrajectoryGeneration(p0_vec, p2_vec, curveIntensity)
        slice_lengths = tg.lengthAlongTheCurveArray(steps)

        mp = self.motionProfile

        a_accel = mp.getMaxAccel()
        a_decel = abs(mp.getMaxDeccel())
        max_velocity = mp.vf

        initial_velocity = mp.vi
        final_velocity = mp.vi

        velocity = initial_velocity
        time_elapsed = 0.0

        total_distance = sum(dx for dx in slice_lengths if dx > 0)
        remaining_distance = total_distance

        if a_accel <= 0 or a_decel <= 0 or max_velocity <= 0:
            # Invalid motion profile parameters
            return True, float("nan")

        for dx in slice_lengths:
            if dx <= 0:
                continue

            braking_distance = 0.0
            if a_decel > 0:
                braking_distance = max(0.0, (velocity ** 2 - final_velocity ** 2) / (2.0 * a_decel))

            if braking_distance >= remaining_distance:
                a = -a_decel
            else:
                if velocity < max_velocity:
                    a = a_accel
                else:
                    a = 0.0

            if abs(a) < 1e-9:
                if velocity <= 0:
                    return True, time_elapsed
                dt = dx / velocity
            else:
                dt = tg.time_Integrand(velocity, dx, a)
                if dt is None:
                    return True, time_elapsed

            time_elapsed += dt
            velocity += a * dt
            remaining_distance -= dx

        if abs(velocity - final_velocity) > 1e-3:
            return True, time_elapsed

        return False, time_elapsed

    def computeTimeWithProfileAndAdjustCurve(self, P0: Pose2D, P2: Pose2D, initialCurveIntensity: float, steps=1000, max_iterations=20, intensity_scale=0.9) -> tuple[float, float]:
        p0_vec = [P0.getX(), P0.getY()]
        p2_vec = [P2.getX(), P2.getY()]

        curveIntensity = initialCurveIntensity
        last_time: float | None = None

        for _ in range(max_iterations):
            broken, total_time = self._simulate_time_for_curve(p0_vec, p2_vec, curveIntensity, steps)
            last_time = total_time
            if not broken:
                return curveIntensity, total_time

            curveIntensity *= intensity_scale

        # If no non-broken solution was found, return the last evaluated time
        return curveIntensity, last_time if last_time is not None else float("nan")
