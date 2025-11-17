import argparse
import csv
import json
from pathlib import Path


def build_config_from_csv(
    csv_path: str,
    vi: float = 0.0,
    vf: float = 5.0,
    max_accel: float = 6.0,
    max_deccel: float = 6.0,
    noccel_t: float = 0.0,
    steps: int = 100,
    default_curve_intensity: float = 5.05,
) -> dict:
    waypoints = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    header = rows[0]

    # Detect header with named columns
    if all(name.lower() in ("x", "y", "heading") for name in header[:3]):
        data_rows = rows[1:]
        x_idx = header.index("x")
        y_idx = header.index("y")
        h_idx = header.index("heading")
        b_idx = header.index("b") if "b" in header else None

        for row in data_rows:
            if len(row) <= max(x_idx, y_idx, h_idx):
                continue
            b_val = default_curve_intensity
            if b_idx is not None and len(row) > b_idx:
                b_val = float(row[b_idx])

            waypoints.append(
                {
                    "x": float(row[x_idx]),
                    "y": float(row[y_idx]),
                    "heading": float(row[h_idx]),
                    "b": b_val,
                }
            )
    else:
        # Positional columns: x, y, heading, optional b in 4th column
        for row in rows:
            if len(row) < 3:
                continue
            x, y, h = row[0], row[1], row[2]
            b_val = default_curve_intensity
            if len(row) > 3:
                b_val = float(row[3])
            waypoints.append(
                {"x": float(x), "y": float(y), "heading": float(h), "b": b_val}
            )

    if len(waypoints) < 2:
        raise ValueError("CSV must contain at least two waypoints to form a segment.")

    segments = []
    for i in range(len(waypoints) - 1):
        b_val = waypoints[i].get("b", default_curve_intensity)
        segments.append(
            {
                "start": waypoints[i],
                "end": waypoints[i + 1],
                "curveIntensity": b_val,
            }
        )

    config = {
        "motionProfile": {
            "vi": vi,
            "vf": vf,
            "maxAccel": max_accel,
            "maxDecel": max_deccel,
            "noccel_t": noccel_t,
        },
        "steps": steps,
        "defaultCurveIntensity": default_curve_intensity,
        "segments": segments,
    }

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trajectory input JSON from a CSV of waypoints."
    )
    parser.add_argument("csv", help="Input CSV file with waypoints (x, y, heading).")
    parser.add_argument("output", help="Output JSON file for main.py.")

    parser.add_argument("--vi", type=float, default=0.0, help="Initial velocity.")
    parser.add_argument("--vf", type=float, default=5.0, help="Max (cruise) velocity (m/s).")
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
    parser.add_argument(
        "--noccel-t",
        type=float,
        default=0.0,
        help="Placeholder cruise time (not used for duration).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Samples per segment.")
    parser.add_argument(
        "--curve-intensity",
        type=float,
        default=5.05,
        help="Default curve intensity (b) for all segments.",
    )

    args = parser.parse_args()

    cfg = build_config_from_csv(
        csv_path=args.csv,
        vi=args.vi,
        vf=args.vf,
        max_accel=args.max_accel,
        max_deccel=args.max_deccel,
        noccel_t=args.noccel_t,
        steps=args.steps,
        default_curve_intensity=args.curve_intensity,
    )

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
