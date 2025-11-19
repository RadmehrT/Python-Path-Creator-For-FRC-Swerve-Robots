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


def _compute_distance_time_profile(
    trajectories: list[dict], section_times: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """
    Build distance- and speed-vs-time profiles using the
    sampled poses on each segment and the per-segment durations.
    """
    if not trajectories or not section_times:
        return [], [], []

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

        # Per-segment arc length built from sampled poses
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

    return times, distances, speeds


def create_visualization_pdf(
    trajectories: list[dict], section_times: list[float], pdf_path: str
) -> None:
    """
    Create a PDF containing:
      - one XY plot per trajectory segment,
      - a combined XY plot showing all segments and waypoints,
      - a distance-over-time plot with speed overlay.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib is not installed; skipping PDF visualization.")
        return

    if not trajectories:
        print("No trajectories to visualize.")
        return

    with PdfPages(pdf_path) as pdf:
        # Individual segment plots: "Trajectory 1, 2, ..."
        for idx, traj in enumerate(trajectories, start=1):
            poses = traj.get("poses", [])
            if not poses:
                continue

            xs = [p["x"] for p in poses]
            ys = [p["y"] for p in poses]

            fig, ax = plt.subplots()
            ax.plot(xs, ys, marker="o", markersize=2)
            ax.set_title(f"Trajectory {idx}: point {idx - 1} to point {idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.axis("equal")

            pdf.savefig(fig)
            plt.close(fig)

        # Combined XY plot of all trajectories, stitched end-to-start
        fig, ax = plt.subplots()

        # Waypoints derived directly from segment start/end (CSV rows)
        waypoints: list[dict] = []
        if trajectories:
            waypoints.append(trajectories[0]["start"])
            for traj in trajectories:
                waypoints.append(traj["end"])

        # Plot each segment curve, but transform its local geometry so that
        # the first and last poses land exactly on the corresponding
        # waypoint coordinates. This keeps the CSV waypoints authoritative.
        for idx, traj in enumerate(trajectories):
            poses = traj.get("poses", [])
            if not poses:
                continue

            if not waypoints or idx + 1 >= len(waypoints):
                continue

            start_wp = waypoints[idx]
            end_wp = waypoints[idx + 1]

            raw_xs = [p["x"] for p in poses]
            raw_ys = [p["y"] for p in poses]

            # Local endpoints from the generated curve
            p0x, p0y = raw_xs[0], raw_ys[0]
            p1x, p1y = raw_xs[-1], raw_ys[-1]

            v_local_x = p1x - p0x
            v_local_y = p1y - p0y
            v_local_len = math.hypot(v_local_x, v_local_y)

            w_global_x = end_wp["x"] - start_wp["x"]
            w_global_y = end_wp["y"] - start_wp["y"]
            w_global_len = math.hypot(w_global_x, w_global_y)

            if v_local_len < 1e-6 or w_global_len < 1e-6:
                # Degenerate segment; just draw a straight line between waypoints.
                ax.plot(
                    [start_wp["x"], end_wp["x"]],
                    [start_wp["y"], end_wp["y"]],
                    linewidth=1.5,
                    label=f"Segment {idx + 1}",
                )
                continue

            # Uniform scale so arc length matches the waypoint chord length
            scale = w_global_len / v_local_len

            # Unit vectors
            ux_local = v_local_x / v_local_len
            uy_local = v_local_y / v_local_len
            ux_global = w_global_x / w_global_len
            uy_global = w_global_y / w_global_len

            # Rotation that maps local direction to global direction
            cos_theta = ux_local * ux_global + uy_local * uy_global
            # Clamp for safety
            cos_theta = max(-1.0, min(1.0, cos_theta))
            sin_theta = ux_local * uy_global - uy_local * ux_global

            def _transform_point(x: float, y: float) -> tuple[float, float]:
                # Shift so p0 is at the origin
                lx = x - p0x
                ly = y - p0y

                # Scale
                lx *= scale
                ly *= scale

                # Rotate
                rx = cos_theta * lx - sin_theta * ly
                ry = sin_theta * lx + cos_theta * ly

                # Translate to the global start waypoint
                gx = start_wp["x"] + rx
                gy = start_wp["y"] + ry
                return gx, gy

            xs, ys = zip(*(_transform_point(x, y) for x, y in zip(raw_xs, raw_ys)))
            ax.plot(xs, ys, linewidth=1.5, label=f"Segment {idx + 1}")

        # Plot the CSV waypoints themselves, in order, as authoritative points.
        if waypoints:
            wx = [p["x"] for p in waypoints]
            wy = [p["y"] for p in waypoints]
            ax.plot(wx, wy, "ro", markersize=4, label="Waypoints")

            for i, p in enumerate(waypoints):
                ax.annotate(
                    str(i),
                    (p["x"], p["y"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

        ax.set_title("All Trajectories Combined (stitched to CSV waypoints)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis("equal")

        pdf.savefig(fig)
        plt.close(fig)

        # Distance and speed versus time
        times, distances, speeds = _compute_distance_time_profile(trajectories, section_times)
        if times and distances:
            fig, ax1 = plt.subplots()
            ax1.plot(times, distances, color="tab:blue", label="Distance")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Distance (m)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(times, speeds, color="tab:orange", label="Speed")
            ax2.set_ylabel("Speed (m/s)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            fig.suptitle("Distance and Speed vs Time")
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

    # 1) Preserve the original output format as myoutput_spliced.json
    with open("myoutput_spliced.json", "w", encoding="utf-8") as f_spliced:
        json.dump(result, f_spliced, indent=2)

    # 2) Build the new flattened-per-trajectory format for my_output.json
    trajectories = result.get("trajectories", [])
    flattened: dict[str, list[dict]] = {}
    for idx, traj in enumerate(trajectories, start=1):
        name = f"traj_{idx}"
        points: list[dict] = []

        start_data = traj.get("start")
        end_data = traj.get("end")
        poses = traj.get("poses", [])

        # First entry: explicit start coordinate
        if start_data is not None:
            points.append(
                {
                    "x": start_data["x"],
                    "y": start_data["y"],
                    "heading": start_data.get("heading", 0.0),
                }
            )

        # All sampled poses along the path
        for p in poses:
            points.append(
                {
                    "x": p["x"],
                    "y": p["y"],
                    "heading": p.get("heading", 0.0),
                }
            )

        # Last entry: explicit end coordinate
        if end_data is not None:
            points.append(
                {
                    "x": end_data["x"],
                    "y": end_data["y"],
                    "heading": end_data.get("heading", 0.0),
                }
            )

        flattened[name] = points

    with open(args.output, "w", encoding="utf-8") as f_flat:
        json.dump(flattened, f_flat, indent=2)

    if trajectories:
        create_visualization_pdf(trajectories, section_times, args.pdf)
        print(f"Saved trajectory visualizations to {args.pdf}")


if __name__ == "__main__":
    main()
