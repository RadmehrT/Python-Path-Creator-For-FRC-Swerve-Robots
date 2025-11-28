import argparse
import json
from pathlib import Path

from generate_input_from_csv import build_config_from_csv
from main import (
    create_visualization_pdf,
    discretize_trajectories,
    flatten_trajectories,
    generate_trajectories_from_config,
    stitch_trajectories_to_waypoints,
)
from trapezoidMotionProfile import trapezoidMotionProfile


def process_csv(
    csv_path: Path,
    output_dir: Path,
    vi: float,
    vf: float,
    max_accel: float,
    max_deccel: float,
    steps: int,
    default_curve_intensity: float,
    sample_period_ms: float,
) -> None:
    """
    Run the full trajectory generation pipeline for a single CSV,
    writing JSON and PDF outputs into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config_from_csv(
        csv_path=str(csv_path),
        vi=vi,
        vf=vf,
        max_accel=max_accel,
        max_deccel=max_deccel,
        steps=steps,
        default_curve_intensity=default_curve_intensity,
    )

    mp_cfg = config.get("motionProfile", {})
    pdf_motion_profile = trapezoidMotionProfile(
        float(mp_cfg.get("vi", vi)),
        float(mp_cfg.get("vf", vf)),
        float(mp_cfg.get("maxAccel", max_accel)),
        float(mp_cfg.get("maxDecel", max_deccel)),
    )

    result, section_times = generate_trajectories_from_config(config)
    trajectories_raw = result.get("trajectories", [])
    stitched_trajectories = stitch_trajectories_to_waypoints(trajectories_raw)

    total_time_all = sum(section_times)
    times_block = {
        "total": total_time_all,
        "sections": [{"section": i + 1, "time": t} for i, t in enumerate(section_times)],
    }

    # Preserve structure with stitched poses
    spliced_path = output_dir / "myoutput_spliced.json"
    with spliced_path.open("w", encoding="utf-8") as f_spliced:
        json.dump({"trajectories": stitched_trajectories}, f_spliced, indent=2)

    # Flatten into traj_n format using stitched poses and include timing summary
    flattened = flatten_trajectories(stitched_trajectories)
    flat_payload = {"times": times_block, "trajectories": flattened}
    flat_path = output_dir / "my_output.json"
    with flat_path.open("w", encoding="utf-8") as f_flat:
        json.dump(flat_payload, f_flat, indent=2)

    # Discretised timeline at fixed sample period
    sample_period_s = max(sample_period_ms / 1000.0, 1e-6)
    discretised_payload = discretize_trajectories(stitched_trajectories, section_times, sample_period_s)
    discretised_path = output_dir / "discretised_output.json"
    with discretised_path.open("w", encoding="utf-8") as f_disc:
        json.dump(discretised_payload, f_disc, indent=2)

    # Visualization PDF for this CSV (stitched geometry)
    if stitched_trajectories:
        pdf_path = output_dir / "trajectories_stitched.pdf"
        create_visualization_pdf(
            stitched_trajectories,
            section_times,
            str(pdf_path),
            already_stitched=True,
            motion_profile=pdf_motion_profile,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner: take multiple waypoint CSV files and "
            "generate trajectories + outputs in numbered subfolders."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="input",
        help="Directory containing one or more CSV waypoint files (default: input).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where per-CSV subfolders (output 1, output 2, ...) will be created (default: output).",
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

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory '{input_dir}' does not exist. Create it and add CSV files.")
        return

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return

    for idx, csv_path in enumerate(csv_files, start=1):
        run_dir = output_root / f"output {idx}"
        print(f"Processing {csv_path} -> {run_dir}")
        process_csv(
            csv_path=csv_path,
            output_dir=run_dir,
            vi=args.vi,
            vf=args.vf,
            max_accel=args.max_accel,
            max_deccel=args.max_deccel,
            steps=args.steps,
            default_curve_intensity=args.curve_intensity,
            sample_period_ms=args.sample_period_ms,
        )


if __name__ == "__main__":
    main()

