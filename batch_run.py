import argparse
import json
from pathlib import Path

from generate_input_from_csv import build_config_from_csv
from main import generate_trajectories_from_config, create_visualization_pdf


def _flatten_trajectories(result: dict) -> dict:
    """
    Build the traj_n-style flattened output structure from the
    result produced by generate_trajectories_from_config.
    """
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

    return flattened


def process_csv(
    csv_path: Path,
    output_dir: Path,
    vi: float,
    vf: float,
    max_accel: float,
    max_deccel: float,
    steps: int,
    default_curve_intensity: float,
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

    result, section_times = generate_trajectories_from_config(config)

    # Preserve original rich structure
    spliced_path = output_dir / "myoutput_spliced.json"
    with spliced_path.open("w", encoding="utf-8") as f_spliced:
        json.dump(result, f_spliced, indent=2)

    # Flatten into traj_n format
    flattened = _flatten_trajectories(result)
    flat_path = output_dir / "my_output.json"
    with flat_path.open("w", encoding="utf-8") as f_flat:
        json.dump(flattened, f_flat, indent=2)

    # Visualization PDF for this CSV
    trajectories = result.get("trajectories", [])
    if trajectories:
        pdf_path = output_dir / "trajectories_stitched.pdf"
        create_visualization_pdf(trajectories, section_times, str(pdf_path))


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
        )


if __name__ == "__main__":
    main()

