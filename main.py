import argparse
import csv
import json
import math
from pathlib import Path

from PathCreator import pathCreator
from Pose2D import Pose2D
from TrajectoryGeneration import TrajectoryGeneration
from trapezoidMotionProfile import trapezoidMotionProfile
from generate_input_from_csv import build_config_from_csv


def pose_from_dict(data: dict) -> Pose2D:
    return Pose2D(data["x"], data["y"], data.get("heading", 0.0))


def pose_to_dict(pose: Pose2D) -> dict:
    return {"x": pose.getX(), "y": pose.getY(), "heading": pose.getHeading()}


def _reflect_poses_across_segment(start: dict, end: dict, poses: list[dict]) -> list[dict]:
    """
    Mirror sampled poses across the line from start->end.
    Keeps endpoints on the line; headings are left unchanged.
    """
    sx, sy = start["x"], start["y"]
    ex, ey = end["x"], end["y"]

    vx = ex - sx
    vy = ey - sy
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return poses

    reflected: list[dict] = []
    for p in poses:
        px, py = p["x"], p["y"]
        # projection of (P - S) onto v
        t = ((px - sx) * vx + (py - sy) * vy) / denom
        qx = sx + t * vx
        qy = sy + t * vy
        rx = 2 * qx - px
        ry = 2 * qy - py
        reflected.append({"x": rx, "y": ry, "heading": p.get("heading", 0.0)})

    return reflected


def _find_best_b_for_segment(
    P0: Pose2D,
    P2: Pose2D,
    initial_b: float,
    vi: float,
    vf: float,
    max_accel: float,
    length_steps: int = 5000,
) -> tuple[float, float]:
    """
    Brute-force search over a small set of b values around the initial guess.
    Chooses the b that minimizes the segment length (equivalently, time).
    """
    p0_vec = [P0.getX(), P0.getY()]
    p2_vec = [P2.getX(), P2.getY()]

    if abs(initial_b) < 1e-6:
        base_candidates = [0.5, 1.0, 1.5]
    else:
        factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        base_candidates = [initial_b * f for f in factors]

    best_b = initial_b
    best_length = float("inf")

    for b in base_candidates:
        tg = TrajectoryGeneration(p0_vec, p2_vec, b)
        low, up = tg.bounds()
        L = tg.lengthAlongTheCurve(low, up, steps=length_steps)

        if L < best_length:
            best_length = L
            best_b = b

    return best_b, best_length


def _compute_global_trapezoid_time(total_length: float, vi: float, vf: float, max_accel: float) -> float:
    """
    Compute total time for a single global trapezoid (0 -> vf -> 0)
    over a given path length with max acceleration max_accel.
    """
    a = max_accel
    if a <= 0 or vf <= 0 or total_length <= 0:
        return 0.0

    # Distance needed to accelerate from 0 to vf and decelerate back to 0
    d_accel = vf * vf / (2.0 * a)
    d_decel = d_accel
    d_min = d_accel + d_decel

    if total_length >= d_min:
        # Full trapezoid: accelerate to vf, cruise, then decelerate
        d_cruise = total_length - d_min
        t_accel = vf / a
        t_decel = vf / a
        t_cruise = d_cruise / vf
        return t_accel + t_cruise + t_decel
    else:
        # Triangular profile: never reach vf, accelerate then decelerate symmetrically
        t_accel = math.sqrt(total_length / a)
        return 2.0 * t_accel


def _solve_time_for_step(vi: float, dx: float, a: float) -> float | None:
    """
    Solve time to traverse distance dx starting at velocity vi with constant accel a.
    Mirrors TrajectoryGeneration.time_Integrand.
    """
    discriminant = vi * vi + 2.0 * a * dx
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t_plus = (-vi + sqrt_disc) / a
    t_minus = (-vi - sqrt_disc) / a

    potential_times = [t for t in (t_plus, t_minus) if t > 0]
    if not potential_times:
        return None

    return min(potential_times)


def _compute_section_times_with_profile(
    segments: list[dict],
    motion_profile: trapezoidMotionProfile,
    final_velocity_target: float | None = None,
) -> tuple[list[float], float, float, bool]:
    """
    Integrate time over all segments using per-slice kinematics.
    Returns (section_times, total_time, final_velocity, broken_flag).
    """
    a_accel = float(motion_profile.getMaxAccel())
    a_decel = float(abs(motion_profile.getMaxDeccel()))
    v_max = float(motion_profile.vf)
    if a_accel <= 0.0 or a_decel <= 0.0 or v_max <= 0.0:
        return [0.0 for _ in segments], 0.0, float(motion_profile.vi), True

    target_final_velocity = float(motion_profile.vi) if final_velocity_target is None else float(final_velocity_target)

    section_times = [0.0 for _ in segments]
    velocity = float(motion_profile.vi)

    total_remaining = 0.0
    for seg in segments:
        total_remaining += sum(dx for dx in seg.get("slices", []) if dx > 0)

    if total_remaining <= 0.0:
        return section_times, 0.0, velocity, True

    remaining_distance = total_remaining

    for seg_idx, seg in enumerate(segments):
        for dx in seg.get("slices", []):
            if dx <= 0:
                continue

            braking_distance = max(0.0, (velocity * velocity - target_final_velocity * target_final_velocity) / (2.0 * a_decel))

            if braking_distance >= remaining_distance:
                a = -a_decel
            else:
                if velocity < v_max:
                    a = a_accel
                else:
                    a = 0.0

            if abs(a) < 1e-9:
                if velocity <= 0.0:
                    return section_times, sum(section_times), velocity, True
                dt = dx / velocity
            else:
                dt = _solve_time_for_step(velocity, dx, a)
                if dt is None:
                    return section_times, sum(section_times), velocity, True

            section_times[seg_idx] += dt
            velocity += a * dt
            remaining_distance -= dx

    return section_times, sum(section_times), velocity, False


def generate_trajectories_from_config(config: dict) -> tuple[dict, list[float]]:
    mp_cfg = config.get("motionProfile", {})
    vi = float(mp_cfg.get("vi", 0.0))
    vf = float(mp_cfg.get("vf", 5.0))
    max_accel = float(mp_cfg.get("maxAccel", 6.0))
    max_deccel = float(mp_cfg.get("maxDecel", max_accel))

    motion_profile = trapezoidMotionProfile(vi, vf, max_accel, max_deccel)
    pc = pathCreator([], motion_profile)

    steps = int(config.get("steps", 100))
    default_b = float(config.get("defaultCurveIntensity", 1.0))
    time_steps = int(config.get("timeIntegrationSteps", 2000))

    trajectories = []
    segment_lengths: list[float] = []
    segment_vectors: list[tuple[float, float]] = []
    segments_for_time: list[dict] = []

    prev_end_heading: float | None = None

    for idx, seg in enumerate(config.get("segments", []), start=1):
        start_data = dict(seg["start"])
        end_data = dict(seg["end"])
        initial_b = float(seg.get("curveIntensity", default_b))

        start_heading = prev_end_heading if prev_end_heading is not None else start_data.get("heading", 0.0)
        end_heading = end_data.get("heading", 0.0)

        # Use continuity for heading at the start of each segment
        start_data["heading"] = start_heading

        P0 = Pose2D(start_data["x"], start_data["y"], start_heading)
        P2 = Pose2D(end_data["x"], end_data["y"], end_heading)

        final_b = initial_b

        poses = pc.GenerateActualPath(P0, P2, final_b, steps=steps)

        p0_vec = [P0.getX(), P0.getY()]
        p2_vec = [P2.getX(), P2.getY()]
        tg = TrajectoryGeneration(p0_vec, p2_vec, final_b)
        low, up = tg.bounds()
        slice_lengths = tg.lengthAlongTheCurveArray(steps=time_steps)
        length = sum(dx for dx in slice_lengths if dx > 0)
        segment_lengths.append(length)
        print(f"[DEBUG] segment {idx} length with b={final_b}: {length}")

        dx = end_data["x"] - start_data["x"]
        dy = end_data["y"] - start_data["y"]
        segment_vectors.append((dx, dy))

        segments_for_time.append(
            {
                "slices": slice_lengths,
                "length": length,
            }
        )

        pose_dicts = [pose_to_dict(p) for p in poses]

        # Mirror segments 2 and 3 across their chord to match desired layout
        if idx in (2, 3):
            pose_dicts = _reflect_poses_across_segment(start_data, end_data, pose_dicts)

        trajectories.append(
            {
                "start": start_data,
                "end": end_data,
                "initialCurveIntensity": initial_b,
                "curveIntensity": final_b,
                "poses": pose_dicts,
            }
        )

        prev_end_heading = end_heading

    section_times, total_time_linear, end_velocity, broken = _compute_section_times_with_profile(
        segments_for_time, motion_profile, final_velocity_target=vi
    )
    total_length = sum(segment_lengths)
    print(f"[DEBUG] total path length = {total_length}")
    if broken:
        total_time_linear = _compute_global_trapezoid_time(total_length, vi, vf, max_accel)
        print(f"[DEBUG] fallback global trapezoid time (translation only) = {total_time_linear}")
        section_times = [0.0 for _ in segment_lengths]
        if total_length > 0:
            for i, seg_len in enumerate(segment_lengths):
                section_times[i] = total_time_linear * (seg_len / total_length)
    else:
        print(f"[DEBUG] integrated translation time (profile-based) = {total_time_linear}")

    base_turn_time_90 = 0.5  # seconds for 90 deg at 5 m/s
    base_turn_speed = 5.0    # m/s reference

    corner_times: list[float] = []
    n_segments = len(segment_lengths)
    for i in range(n_segments - 1):
        vx1, vy1 = segment_vectors[i]
        vx2, vy2 = segment_vectors[i + 1]

        mag1 = math.hypot(vx1, vy1)
        mag2 = math.hypot(vx2, vy2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            corner_times.append(0.0)
            continue

        dot = vx1 * vx2 + vy1 * vy2
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 1e-3:
            corner_times.append(0.0)
            continue

        angle_clamped = min(angle_deg, 90.0)
        speed_scale = vf / base_turn_speed if base_turn_speed > 0 else 1.0
        t_corner = base_turn_time_90 * (angle_clamped / 90.0) * speed_scale
        corner_times.append(t_corner)
        print(f"[DEBUG] corner {i+1}: angle={angle_deg:.2f} deg, extra turn time={t_corner:.3f} s")

    total_turn_time = sum(corner_times)
    total_time = total_time_linear + total_turn_time

    section_times: list[float] = [0.0 for _ in segment_lengths]
    if total_length > 0:
        for i, seg_len in enumerate(segment_lengths):
            section_times[i] = total_time_linear * (seg_len / total_length)

    per_segment_turn = [0.0 for _ in segment_lengths]
    for i, t_corner in enumerate(corner_times):
        per_segment_turn[i] += 0.5 * t_corner
        per_segment_turn[i + 1] += 0.5 * t_corner

    for i in range(len(section_times)):
        section_times[i] += per_segment_turn[i]

    return {"trajectories": trajectories}, section_times


def _compute_distance_time_profile(
    trajectories: list[dict], section_times: list[float]
) -> tuple[list[float], list[float], list[float], float]:
    """
    Build distance- and speed-vs-time profiles using the
    sampled poses on each segment and the per-segment durations.
    Returns the cumulative path length for downstream visualisations.
    """
    if not trajectories or not section_times:
        return [], [], [], 0.0

    n_segments = min(len(trajectories), len(section_times))

    times: list[float] = []
    distances: list[float] = []

    t_global = 0.0
    s_global = 0.0

    for seg_idx in range(n_segments):
        traj = trajectories[seg_idx]
        poses = traj.get("poses", [])
        if not poses:
            t_global += float(section_times[seg_idx])
            continue

        seg_distances = [0.0]
        seg_len = 0.0
        for i in range(1, len(poses)):
            x0, y0 = poses[i - 1]["x"], poses[i - 1]["y"]
            x1, y1 = poses[i]["x"], poses[i]["y"]
            ds = math.hypot(x1 - x0, y1 - y0)
            seg_len += ds
            seg_distances.append(seg_len)

        T = float(section_times[seg_idx])
        if seg_len <= 0.0 or T <= 0.0:
            for _ in seg_distances:
                times.append(t_global)
                distances.append(s_global)
        else:
            for local_s in seg_distances:
                frac = local_s / seg_len
                times.append(t_global + frac * T)
                distances.append(s_global + local_s)

        t_global += T
        s_global += seg_len

    if not times:
        return [], [], []

    speeds: list[float] = [0.0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        ds = distances[i] - distances[i - 1]
        if dt <= 1e-6:
            speeds.append(speeds[-1])
        else:
            speeds.append(ds / dt)

    return times, distances, speeds, s_global


def _build_trapezoid_velocity_profile(
    total_distance: float, motion_profile: trapezoidMotionProfile | None
) -> tuple[list[float], list[float], dict[str, dict[str, float]]]:
    """
    Construct the ideal trapezoid (or triangular) velocity profile based on the
    configured motion profile limits and the provided total travel distance.
    Returns the sampled velocity profile alongside metadata describing each stage.
    """
    if motion_profile is None or total_distance <= 1e-9:
        return [], [], {}

    a_accel = float(motion_profile.getMaxAccel())
    a_decel = float(abs(motion_profile.getMaxDeccel()))
    if a_accel <= 0.0 or a_decel <= 0.0:
        return [], [], {}

    v_start = float(motion_profile.vi)
    v_end = float(motion_profile.vi)
    v_max = float(motion_profile.vf)
    v_max = max(v_max, v_start, v_end)

    def _segment_distance(v0: float, v1: float, accel: float) -> float:
        if accel <= 0.0 or v1 <= v0:
            return 0.0
        return (v1 * v1 - v0 * v0) / (2.0 * accel)

    v_peak = v_max
    d_accel = _segment_distance(v_start, v_peak, a_accel)
    d_decel = _segment_distance(v_end, v_peak, a_decel)

    if total_distance < d_accel + d_decel - 1e-9:
        denom = (1.0 / (2.0 * a_accel)) + (1.0 / (2.0 * a_decel))
        if denom <= 0.0:
            return [], []
        numer = total_distance + (v_start * v_start) / (2.0 * a_accel) + (v_end * v_end) / (2.0 * a_decel)
        v_peak_sq = max(v_start * v_start, v_end * v_end, numer / denom)
        v_peak = min(v_max, math.sqrt(max(0.0, v_peak_sq)))
        d_accel = _segment_distance(v_start, v_peak, a_accel)
        d_decel = _segment_distance(v_end, v_peak, a_decel)

    d_cruise = max(0.0, total_distance - d_accel - d_decel)

    def _segment_time(v0: float, v1: float, accel: float) -> float:
        if accel <= 0.0 or v1 <= v0:
            return 0.0
        return (v1 - v0) / accel

    t_accel = _segment_time(v_start, v_peak, a_accel)
    t_decel = _segment_time(v_end, v_peak, a_decel)
    t_cruise = d_cruise / v_peak if v_peak > 1e-9 else 0.0

    total_time = t_accel + t_cruise + t_decel
    if total_time <= 0.0:
        if abs(v_start) < 1e-9:
            return [], [], {}
        cruise_time = total_distance / abs(v_start)
        return [0.0, cruise_time], [v_start, v_start], {}

    times = [0.0]
    velocities = [v_start]
    current_time = 0.0

    if t_accel > 1e-9:
        current_time += t_accel
        times.append(current_time)
        velocities.append(v_peak)

    if t_cruise > 1e-9:
        current_time += t_cruise
        times.append(current_time)
        velocities.append(v_peak)

    if t_decel > 1e-9:
        current_time += t_decel
        times.append(current_time)
        velocities.append(v_end)
    elif abs(times[-1] - total_time) > 1e-9:
        current_time = total_time
        times.append(current_time)
        velocities.append(v_end)

    stage_info: dict[str, dict[str, float]] = {}
    if t_accel > 1e-9:
        stage_info["accel"] = {"start": 0.0, "end": t_accel, "accel": a_accel}
    if t_cruise > 1e-9:
        stage_info["cruise"] = {"start": t_accel, "end": t_accel + t_cruise}
    if t_decel > 1e-9:
        stage_info["decel"] = {
            "start": total_time - t_decel,
            "end": total_time,
            "accel": -a_decel,
        }

    return times, velocities, stage_info


def stitch_trajectories_to_waypoints(trajectories: list[dict]) -> list[dict]:
    """
    Rescale/rotate each segment's sampled poses so the endpoints land exactly on
    the CSV waypoints (matches the stitched PDF view).
    """
    if not trajectories:
        return []

    waypoints: list[dict] = []
    first_start = trajectories[0].get("start")
    if first_start:
        waypoints.append(first_start)
    for traj in trajectories:
        end_pt = traj.get("end")
        if end_pt:
            waypoints.append(end_pt)

    stitched: list[dict] = []

    for idx, traj in enumerate(trajectories):
        poses = traj.get("poses", [])
        if not poses:
            stitched.append(traj)
            continue

        if idx >= len(waypoints) - 1:
            stitched.append(traj)
            continue

        start_wp = waypoints[idx]
        end_wp = waypoints[idx + 1]
        if not start_wp or not end_wp:
            stitched.append(traj)
            continue

        raw_xs = [p["x"] for p in poses]
        raw_ys = [p["y"] for p in poses]
        headings = [p.get("heading", 0.0) for p in poses]

        if not raw_xs or not raw_ys:
            stitched.append(traj)
            continue

        local_dx = raw_xs[-1] - raw_xs[0]
        local_dy = raw_ys[-1] - raw_ys[0]
        global_dx = end_wp["x"] - start_wp["x"]
        global_dy = end_wp["y"] - start_wp["y"]

        if local_dx * global_dx + local_dy * global_dy < 0:
            raw_xs = list(reversed(raw_xs))
            raw_ys = list(reversed(raw_ys))
            headings = list(reversed(headings))

        p0x, p0y = raw_xs[0], raw_ys[0]
        p1x, p1y = raw_xs[-1], raw_ys[-1]

        v_local_x = p1x - p0x
        v_local_y = p1y - p0y
        v_local_len = math.hypot(v_local_x, v_local_y)

        w_global_x = global_dx
        w_global_y = global_dy
        w_global_len = math.hypot(w_global_x, w_global_y)

        stitched_poses: list[dict] = []

        if v_local_len < 1e-6 or w_global_len < 1e-6:
            stitched_poses.append(
                {"x": start_wp["x"], "y": start_wp["y"], "heading": headings[0]}
            )
            stitched_poses.append(
                {"x": end_wp["x"], "y": end_wp["y"], "heading": headings[-1]}
            )
        else:
            scale = w_global_len / v_local_len

            ux_local = v_local_x / v_local_len
            uy_local = v_local_y / v_local_len
            ux_global = w_global_x / w_global_len
            uy_global = w_global_y / w_global_len

            cos_theta = max(-1.0, min(1.0, ux_local * ux_global + uy_local * uy_global))
            sin_theta = ux_local * uy_global - uy_local * ux_global

            for x, y, h in zip(raw_xs, raw_ys, headings):
                lx = (x - p0x) * scale
                ly = (y - p0y) * scale

                rx = cos_theta * lx - sin_theta * ly
                ry = sin_theta * lx + cos_theta * ly

                gx = start_wp["x"] + rx
                gy = start_wp["y"] + ry

                stitched_poses.append({"x": gx, "y": gy, "heading": h})

        stitched.append({**traj, "poses": stitched_poses})

    return stitched


def flatten_trajectories(trajectories: list[dict]) -> dict[str, list[dict]]:
    """
    Flatten trajectories into traj_n arrays used by my_output.json.
    """
    flattened: dict[str, list[dict]] = {}

    for idx, traj in enumerate(trajectories, start=1):
        name = f"traj_{idx}"
        points: list[dict] = []

        start_data = traj.get("start")
        end_data = traj.get("end")
        poses = traj.get("poses", [])

        if start_data is not None:
            points.append(
                {
                    "x": start_data["x"],
                    "y": start_data["y"],
                    "heading": start_data.get("heading", 0.0),
                }
            )

        for p in poses:
            points.append(
                {
                    "x": p["x"],
                    "y": p["y"],
                    "heading": p.get("heading", 0.0),
                }
            )

        if end_data is not None:
            points.append(
                {
                    "x": end_data["x"],
                    "y": end_data["y"],
                    "heading": end_data.get("heading", 0.0),
                }
            )

        flattened[name] = points

    return flattened


def _cumulative_distances_for_poses(poses: list[dict]) -> tuple[list[float], float]:
    """
    Return cumulative distances between consecutive poses in a segment.
    """
    if not poses:
        return [], 0.0

    distances = [0.0]
    total = 0.0
    for i in range(1, len(poses)):
        x0, y0 = poses[i - 1]["x"], poses[i - 1]["y"]
        x1, y1 = poses[i]["x"], poses[i]["y"]
        total += math.hypot(x1 - x0, y1 - y0)
        distances.append(total)

    return distances, total


def _interpolate_pose_for_distance(poses: list[dict], cumulative: list[float], target_distance: float) -> dict:
    """
    Linearly interpolate pose (x, y, heading) at a target distance along a segment.
    """
    if not poses:
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    if not cumulative:
        cumulative = [0.0 for _ in poses]

    if target_distance <= 0.0:
        p = poses[0]
        return {"x": p["x"], "y": p["y"], "heading": p.get("heading", 0.0)}

    max_distance = cumulative[-1] if cumulative else 0.0
    if target_distance >= max_distance:
        p = poses[-1]
        return {"x": p["x"], "y": p["y"], "heading": p.get("heading", 0.0)}

    for idx in range(1, len(cumulative)):
        if target_distance <= cumulative[idx]:
            prev_dist = cumulative[idx - 1]
            span = cumulative[idx] - prev_dist
            if span <= 1e-9:
                p = poses[idx]
                return {"x": p["x"], "y": p["y"], "heading": p.get("heading", 0.0)}

            ratio = (target_distance - prev_dist) / span
            p0 = poses[idx - 1]
            p1 = poses[idx]
            x = p0["x"] + ratio * (p1["x"] - p0["x"])
            y = p0["y"] + ratio * (p1["y"] - p0["y"])
            h0 = p0.get("heading", 0.0)
            h1 = p1.get("heading", 0.0)
            heading = h0 + ratio * (h1 - h0)
            return {"x": x, "y": y, "heading": heading}

    p = poses[-1]
    return {"x": p["x"], "y": p["y"], "heading": p.get("heading", 0.0)}


def discretize_trajectories(
    trajectories: list[dict], section_times: list[float], sample_period_s: float = 0.02
) -> dict:
    """
    Produce a uniformly-timed sample set across stitched trajectories.
    """
    sample_period_s = max(sample_period_s, 1e-6)

    if not trajectories or not section_times:
        return {
            "dt": sample_period_s,
            "totalTime": 0.0,
            "totalPathLength": 0.0,
            "numSamples": 0,
            "samples": [],
        }

    n_segments = min(len(trajectories), len(section_times))
    segments: list[dict] = []
    time_cursor = 0.0
    distance_cursor = 0.0

    for i in range(n_segments):
        traj = trajectories[i]
        poses = traj.get("poses", [])
        cumulative_dists, seg_len = _cumulative_distances_for_poses(poses)
        seg_time = float(section_times[i])

        segments.append(
            {
                "poses": poses,
                "cumulative": cumulative_dists,
                "length": seg_len,
                "t_start": time_cursor,
                "t_end": time_cursor + seg_time,
                "dist_start": distance_cursor,
                "start": traj.get("start"),
                "end": traj.get("end"),
            }
        )

        time_cursor += seg_time
        distance_cursor += seg_len

    total_time = time_cursor
    total_path = distance_cursor

    if total_time <= 0.0:
        return {
            "dt": sample_period_s,
            "totalTime": 0.0,
            "totalPathLength": total_path,
            "numSamples": 0,
            "samples": [],
        }

    num_steps = int(math.floor(total_time / sample_period_s + 1e-9))
    time_samples = [i * sample_period_s for i in range(num_steps + 1)]
    if total_time - time_samples[-1] > 1e-9:
        time_samples.append(total_time)

    samples: list[dict] = []
    seg_idx = 0

    for t in time_samples:
        while seg_idx < len(segments) - 1 and t > segments[seg_idx]["t_end"]:
            seg_idx += 1

        seg = segments[seg_idx]
        seg_duration = seg["t_end"] - seg["t_start"]
        poses = seg["poses"]
        cumulative = seg["cumulative"]
        seg_len = seg["length"]

        if seg_duration <= 0.0 or seg_len <= 0.0 or not poses:
            fallback = (
                poses[0]
                if poses
                else seg["start"]
                or seg["end"]
                or {"x": 0.0, "y": 0.0, "heading": 0.0}
            )
            samples.append(
                {
                    "time": t,
                    "x": fallback["x"],
                    "y": fallback["y"],
                    "heading": fallback.get("heading", 0.0),
                    "distance": seg["dist_start"],
                    "segment": seg_idx + 1,
                }
            )
            continue

        frac = (t - seg["t_start"]) / seg_duration
        frac = max(0.0, min(1.0, frac))
        local_distance = frac * seg_len

        pose = _interpolate_pose_for_distance(poses, cumulative, local_distance)

        samples.append(
            {
                "time": t,
                "x": pose["x"],
                "y": pose["y"],
                "heading": pose["heading"],
                "distance": seg["dist_start"] + local_distance,
                "segment": seg_idx + 1,
            }
        )

    return {
        "dt": sample_period_s,
        "totalTime": total_time,
        "totalPathLength": total_path,
        "numSamples": len(samples),
        "samples": samples,
    }


def _write_spliced_csv(trajectories: list[dict], csv_path: Path) -> None:
    """
    Emit stitched trajectories to CSV rows (trajectory, index, pose type, pose data).
    """
    fieldnames = [
        "trajectory",
        "point_index",
        "point_type",
        "x",
        "y",
        "heading",
        "initialCurveIntensity",
        "curveIntensity",
        "b",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for traj_idx, traj in enumerate(trajectories, start=1):
            curve_intensity = traj.get("curveIntensity")
            initial_curve = traj.get("initialCurveIntensity")
            point_idx = 0

            def emit(point: dict, point_type: str, idx: int) -> None:
                writer.writerow(
                    {
                        "trajectory": traj_idx,
                        "point_index": idx,
                        "point_type": point_type,
                        "x": point.get("x"),
                        "y": point.get("y"),
                        "heading": point.get("heading", 0.0),
                        "initialCurveIntensity": initial_curve,
                        "curveIntensity": curve_intensity,
                        "b": point.get("b"),
                    }
                )

            start = traj.get("start")
            if start:
                emit(start, "start", point_idx)
                point_idx += 1

            for pose in traj.get("poses", []):
                emit(pose, "pose", point_idx)
                point_idx += 1

            end = traj.get("end")
            if end:
                emit(end, "end", point_idx)


def _write_flattened_csv(flat_payload: dict, csv_path: Path, times_csv_path: Path) -> None:
    """
    Convert the flattened payload (my_output.json) into CSVs.
    """
    trajectories = flat_payload.get("trajectories", {})
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trajectory", "point_index", "x", "y", "heading"])
        writer.writeheader()
        if isinstance(trajectories, dict):
            for traj_name, points in trajectories.items():
                for idx, pt in enumerate(points):
                    writer.writerow(
                        {
                            "trajectory": traj_name,
                            "point_index": idx,
                            "x": pt.get("x"),
                            "y": pt.get("y"),
                            "heading": pt.get("heading", 0.0),
                        }
                    )

    times = flat_payload.get("times", {})
    sections = times.get("sections", [])
    with times_csv_path.open("w", newline="", encoding="utf-8") as f_times:
        writer = csv.DictWriter(f_times, fieldnames=["section", "time"])
        writer.writeheader()
        if isinstance(sections, list):
            for sec in sections:
                writer.writerow({"section": sec.get("section"), "time": sec.get("time")})
        if "total" in times:
            writer.writerow({"section": "total", "time": times.get("total")})


def _write_discretised_csv(discretised_payload: dict, csv_path: Path) -> None:
    """
    Convert discretised timeline samples into a CSV file.
    """
    samples = discretised_payload.get("samples", [])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", "x", "y", "heading", "distance", "segment"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(samples, list):
            for sample in samples:
                writer.writerow(
                    {
                        "time": sample.get("time"),
                        "x": sample.get("x"),
                        "y": sample.get("y"),
                        "heading": sample.get("heading"),
                        "distance": sample.get("distance"),
                        "segment": sample.get("segment"),
                    }
                )


def write_output_csvs(
    stitched_trajectories: list[dict],
    flat_payload: dict,
    discretised_payload: dict,
    output_dir: Path,
    flat_json_name: str,
) -> None:
    """
    Write CSV siblings for the standard JSON outputs.
    """
    _write_spliced_csv(stitched_trajectories, output_dir / "myoutput_spliced.csv")

    flat_stem = Path(flat_json_name).stem
    _write_flattened_csv(
        flat_payload,
        output_dir / f"{flat_stem}.csv",
        output_dir / f"{flat_stem}_times.csv",
    )

    _write_discretised_csv(discretised_payload, output_dir / "discretised_output.csv")


def _next_output_dir(base_dir: Path) -> Path:
    """
    Find the next numbered subdirectory under base_dir named 'output N'.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        if not name.lower().startswith("output "):
            continue
        try:
            idx = int(name.split(" ", 1)[1])
            max_idx = max(max_idx, idx)
        except (IndexError, ValueError):
            continue
    next_idx = max_idx + 1
    return base_dir / f"output {next_idx}"


def create_visualization_pdf(
    trajectories: list[dict],
    section_times: list[float],
    pdf_path: str,
    already_stitched: bool = False,
    motion_profile: trapezoidMotionProfile | None = None,
) -> None:
    """
    Create a PDF containing:
      - one XY plot per trajectory segment,
      - a combined XY plot showing all segments and waypoints,
      - a distance-over-time plot with speed overlay,
      - the trapezoid motion profile (velocity vs time) derived from the configured limits.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib is not installed; skipping PDF visualization.")
        return

    plt.rcParams["figure.figsize"] = (10.5, 5.5)

    if not trajectories:
        print("No trajectories to visualize.")
        return

    stitched_trajs = trajectories if already_stitched else stitch_trajectories_to_waypoints(trajectories)

    with PdfPages(pdf_path) as pdf:
        for idx, traj in enumerate(stitched_trajs, start=1):
            poses = traj.get("poses", [])
            if not poses:
                continue

            xs = [p["x"] for p in poses]
            ys = [p["y"] for p in poses]

            fig, ax = plt.subplots()
            ax.plot(xs, ys, marker="o", markersize=2)
            ax.set_title(f"Trajectory {idx}: point {idx - 1} to point {idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y", rotation=0, labelpad=25)
            ax.grid(True)
            ax.axis("equal")

            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots()

        waypoints: list[dict] = []
        if stitched_trajs:
            waypoints.append(stitched_trajs[0]["start"])
            for traj in stitched_trajs:
                waypoints.append(traj["end"])

        for idx, traj in enumerate(stitched_trajs):
            poses = traj.get("poses", [])
            if not poses:
                continue

            xs = [p["x"] for p in poses]
            ys = [p["y"] for p in poses]
            ax.plot(xs, ys, linewidth=1.5, label=f"Segment {idx + 1}")

        if waypoints:
            wx = [p["x"] for p in waypoints]
            wy = [p["y"] for p in waypoints]
            ax.plot(wx, wy, "o", color="red", markersize=5, label="Waypoints")

            for i, p in enumerate(waypoints):
                annotate_style: dict = {
                    "fontsize": 10,
                    "fontweight": "bold",
                    "color": "black",
                    "bbox": {
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "black",
                        "lw": 0.5,
                    },
                }
                ax.annotate(
                    str(i),
                    (p["x"], p["y"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    **annotate_style,
                )

        ax.set_title("All Trajectories Combined (stitched to CSV waypoints)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y", rotation=0, labelpad=25)
        ax.grid(True)
        ax.axis("equal")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=1,
            fontsize=9,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        times, distances, speeds, total_distance = _compute_distance_time_profile(stitched_trajs, section_times)
        if times and distances:
            fig, ax1 = plt.subplots()
            ax1.plot(times, distances, color="tab:blue", label="Distance")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Distance (m)", color="tab:blue", rotation=0, labelpad=45)
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(times, speeds, color="tab:orange", label="Speed")
            ax2.set_ylabel("Speed (m/s)", color="tab:orange", rotation=0, labelpad=45)
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            fig.suptitle("Distance and Speed vs Time")
            pdf.savefig(fig)
            plt.close(fig)

        profile_times, profile_velocities, profile_stages = _build_trapezoid_velocity_profile(
            total_distance, motion_profile
        )
        if profile_times and profile_velocities:
            fig = plt.figure(figsize=(11.5, 5.0))
            gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.35)
            ax = fig.add_subplot(gs[0, 0])
            info_ax = fig.add_subplot(gs[0, 1])
            info_ax.axis("off")
            ax.plot(profile_times, profile_velocities, color="tab:green")
            ax.set_title("Trapezoidal Motion Profile")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (m/s)", rotation=0, labelpad=30)
            ax.grid(True)

            annotation_lines: list[str] = []
            accel_section: list[str] = []
            accel_info = profile_stages.get("accel")
            if accel_info:
                accel_section.append(f"Acceleration: {accel_info['accel']:.2f} m/s/s")

            cruise_info = profile_stages.get("cruise")
            decel_info = profile_stages.get("decel")
            if decel_info:
                accel_section.append(f"Deceleration: {abs(decel_info['accel']):.2f} m/s/s")

            if accel_section:
                annotation_lines.append("Acceleration Magnitudes (m/s/s)")
                annotation_lines.append("------------------------------")
                for entry in accel_section:
                    annotation_lines.append(f"  - {entry}")

            max_velocity = motion_profile.vf if motion_profile is not None else None
            if max_velocity is not None:
                if annotation_lines:
                    annotation_lines.append("")  # spacer
                annotation_lines.append("Maximum Velocity (m/s)")
                annotation_lines.append("----------------------")
                annotation_lines.append(f"  - {max_velocity:.2f} m/s")

            time_section: list[str] = []
            if accel_info:
                time_section.append(
                    f"Acceleration: t={accel_info['start']:.2f}-{accel_info['end']:.2f}s"
                )
            if cruise_info:
                time_section.append(
                    f"Cruise: t={cruise_info['start']:.2f}-{cruise_info['end']:.2f}s"
                )
            if decel_info:
                time_section.append(
                    f"Deceleration: t={decel_info['start']:.2f}-{decel_info['end']:.2f}s"
                )

            if time_section:
                if annotation_lines:
                    annotation_lines.append("")  # spacer between sections
                annotation_lines.append("Stage Timing (s)")
                annotation_lines.append("------------------------------")
                for entry in time_section:
                    annotation_lines.append(f"  - {entry}")

            if annotation_lines:
                details_text = "\n".join(annotation_lines)
                info_ax.text(
                    0.0,
                    1.0,
                    details_text,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "lw": 0.5},
                )

            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trajectories from a CSV of waypoints."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="Math IA Waypoints list - Sheet1.csv",
        help="Input CSV file with waypoints (x, y, heading).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="my_output.json",
        help="Output JSON file for generated trajectories.",
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        const="trajectories.pdf",
        default="trajectories.pdf",
        help="Output PDF file for trajectory visualizations.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where numbered outputs will be written (default: output).",
    )

    parser.add_argument("--vi", type=float, default=0.0, help="Initial velocity.")
    parser.add_argument(
        "--vf",
        type=float,
        default=5.0,
        help="Max (cruise) velocity (m/s).",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        default=6.0,
        help="Maximum acceleration magnitude (m/s^2).",
    )
    parser.add_argument(
        "--max-deccel",
        type=float,
        default=6.0,
        help="Maximum deceleration magnitude (m/s^2).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Samples per segment.")
    parser.add_argument(
        "--curve-intensity",
        type=float,
        default=5.05,
        help="Default curve intensity (b) for all segments.",
    )
    parser.add_argument(
        "--sample-period-ms",
        type=float,
        default=20.0,
        help="Output period in milliseconds for discretised_output.json (default: 20 ms).",
    )

    args = parser.parse_args()

    config = build_config_from_csv(
        csv_path=args.csv,
        vi=args.vi,
        vf=args.vf,
        max_accel=args.max_accel,
        max_deccel=args.max_deccel,
        steps=args.steps,
        default_curve_intensity=args.curve_intensity,
    )

    mp_cfg = config.get("motionProfile", {})
    pdf_motion_profile = trapezoidMotionProfile(
        float(mp_cfg.get("vi", args.vi)),
        float(mp_cfg.get("vf", args.vf)),
        float(mp_cfg.get("maxAccel", args.max_accel)),
        float(mp_cfg.get("maxDecel", args.max_deccel)),
    )

    base_output_dir = Path(args.output_dir)
    run_output_dir = _next_output_dir(base_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    result, section_times = generate_trajectories_from_config(config)
    trajectories_raw = result.get("trajectories", [])
    stitched_trajectories = stitch_trajectories_to_waypoints(trajectories_raw)

    print("section time")
    total_time_all = 0.0
    section_time_labels: list[dict] = []
    for idx, t in enumerate(section_times, start=1):
        print(f"section {idx}: {t}")
        total_time_all += t
        section_time_labels.append({"section": idx, "time": t})
    print(f"total time: {total_time_all}")
    times_block = {"total": total_time_all, "sections": section_time_labels}

    # Resolve output paths inside the numbered output directory
    spliced_path = run_output_dir / "myoutput_spliced.json"
    flat_path = run_output_dir / Path(args.output).name
    pdf_path = run_output_dir / Path(args.pdf).name
    discretised_path = run_output_dir / "discretised_output.json"

    # 1) stitched structure for downstream use
    stitched_result = {"trajectories": stitched_trajectories}
    with spliced_path.open("w", encoding="utf-8") as f_spliced:
        json.dump(stitched_result, f_spliced, indent=2)

    # 2) flattened stitched poses with timing summary
    flattened = flatten_trajectories(stitched_trajectories)
    output_payload = {"times": times_block, "trajectories": flattened}
    with flat_path.open("w", encoding="utf-8") as f_flat:
        json.dump(output_payload, f_flat, indent=2)

    # 3) discretised output sampled every sample_period_ms (default 20 ms)
    sample_period_s = max(args.sample_period_ms / 1000.0, 1e-6)
    discretised_payload = discretize_trajectories(stitched_trajectories, section_times, sample_period_s)
    with discretised_path.open("w", encoding="utf-8") as f_disc:
        json.dump(discretised_payload, f_disc, indent=2)

    write_output_csvs(
        stitched_trajectories,
        output_payload,
        discretised_payload,
        run_output_dir,
        Path(args.output).name,
    )

    if stitched_trajectories:
        create_visualization_pdf(
            stitched_trajectories,
            section_times,
            str(pdf_path),
            already_stitched=True,
            motion_profile=pdf_motion_profile,
        )
        print(f"Saved trajectory visualizations to {pdf_path}")

    print(f"Outputs written to {run_output_dir}")


if __name__ == "__main__":
    main()
