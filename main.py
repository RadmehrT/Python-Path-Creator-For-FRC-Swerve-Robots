import argparse
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
        length = tg.lengthAlongTheCurve(low, up, steps=10000)
        segment_lengths.append(length)
        print(f"[DEBUG] segment {idx} length with b={final_b}: {length}")

        dx = end_data["x"] - start_data["x"]
        dy = end_data["y"] - start_data["y"]
        segment_vectors.append((dx, dy))

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

    total_length = sum(segment_lengths)
    print(f"[DEBUG] total path length = {total_length}")
    total_time_linear = _compute_global_trapezoid_time(total_length, vi, vf, max_accel)
    print(f"[DEBUG] global trapezoid time (translation only) = {total_time_linear}")

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
    trajectories: list[dict], section_times: list[float], pdf_path: str, already_stitched: bool = False
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
            ax.set_ylabel("Y")
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
        ax.set_ylabel("Y")
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

        times, distances, speeds = _compute_distance_time_profile(stitched_trajs, section_times)
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

    # 1) stitched structure for downstream use
    stitched_result = {"trajectories": stitched_trajectories}
    with spliced_path.open("w", encoding="utf-8") as f_spliced:
        json.dump(stitched_result, f_spliced, indent=2)

    # 2) flattened stitched poses with timing summary
    flattened = flatten_trajectories(stitched_trajectories)
    output_payload = {"times": times_block, "trajectories": flattened}
    with flat_path.open("w", encoding="utf-8") as f_flat:
        json.dump(output_payload, f_flat, indent=2)

    if stitched_trajectories:
        create_visualization_pdf(stitched_trajectories, section_times, str(pdf_path), already_stitched=True)
        print(f"Saved trajectory visualizations to {pdf_path}")

    print(f"Outputs written to {run_output_dir}")


if __name__ == "__main__":
    main()
