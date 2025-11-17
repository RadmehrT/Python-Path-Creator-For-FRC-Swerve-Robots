import argparse
import json
import math

from PathCreator import pathCreator
from Pose2D import Pose2D
from TrajectoryGeneration import TrajectoryGeneration
from trapezoidMotionProfile import trapezoidMotionProfile
from generate_input_from_csv import build_config_from_csv


def pose_from_dict(data: dict) -> Pose2D:
    return Pose2D(data["x"], data["y"], data.get("heading", 0.0))


def pose_to_dict(pose: Pose2D) -> dict:
    return {"x": pose.getX(), "y": pose.getY(), "heading": pose.getHeading()}


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
        # total_length = a * t_accel^2  =>  t_accel = sqrt(total_length / a)
        t_accel = math.sqrt(total_length / a)
        return 2.0 * t_accel


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

    trajectories = []
    segment_lengths: list[float] = []
    segment_vectors: list[tuple[float, float]] = []

    for idx, seg in enumerate(config.get("segments", []), start=1):
        start_data = seg["start"]
        end_data = seg["end"]
        initial_b = float(seg.get("curveIntensity", default_b))

        P0 = pose_from_dict(start_data)
        P2 = pose_from_dict(end_data)

        # Use the provided curve intensity directly for geometry
        final_b = initial_b

        # Generate poses along this segment with the chosen intensity
        poses = pc.GenerateActualPath(P0, P2, final_b, steps=steps)

        # Compute arc length of this segment for global time calculation
        p0_vec = [P0.getX(), P0.getY()]
        p2_vec = [P2.getX(), P2.getY()]
        tg = TrajectoryGeneration(p0_vec, p2_vec, final_b)
        low, up = tg.bounds()
        length = tg.lengthAlongTheCurve(low, up, steps=10000)
        segment_lengths.append(length)
        print(f"[DEBUG] segment {idx} length with b={final_b}: {length}")

        # straight-line vector for heading-change estimation
        dx = end_data["x"] - start_data["x"]
        dy = end_data["y"] - start_data["y"]
        segment_vectors.append((dx, dy))

        trajectories.append(
            {
                "start": start_data,
                "end": end_data,
                "initialCurveIntensity": initial_b,
                "curveIntensity": final_b,
                "poses": [pose_to_dict(p) for p in poses],
            }
        )

    total_length = sum(segment_lengths)
    print(f"[DEBUG] total path length = {total_length}")
    total_time_linear = _compute_global_trapezoid_time(total_length, vi, vf, max_accel)
    print(f"[DEBUG] global trapezoid time (translation only) = {total_time_linear}")

    # Extra time due to heading changes at segment junctions.
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

    # Distribute translation time by segment length
    section_times: list[float] = [0.0 for _ in segment_lengths]
    if total_length > 0:
        for i, seg_len in enumerate(segment_lengths):
            section_times[i] = total_time_linear * (seg_len / total_length)

    # Distribute heading-change time: split each corner equally between its two adjacent segments
    per_segment_turn = [0.0 for _ in segment_lengths]
    for i, t_corner in enumerate(corner_times):
        per_segment_turn[i] += 0.5 * t_corner
        per_segment_turn[i + 1] += 0.5 * t_corner

    for i in range(len(section_times)):
        section_times[i] += per_segment_turn[i]

    return {"trajectories": trajectories}, section_times


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

    result, section_times = generate_trajectories_from_config(config)

    print("section time")
    total_time_all = 0.0
    for idx, t in enumerate(section_times, start=1):
        print(f"section {idx}: {t}")
        total_time_all += t
    print(f"total time: {total_time_all}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
