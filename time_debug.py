import math

from generate_input_from_csv import build_config_from_csv
from TrajectoryGeneration import TrajectoryGeneration
from main import _compute_global_trapezoid_time


cfg = build_config_from_csv("Math IA Waypoints list - Sheet1.csv")

mp_cfg = cfg.get("motionProfile", {})
vi = float(mp_cfg.get("vi", 0.0))
vf = float(mp_cfg.get("vf", 5.0))
max_accel = float(mp_cfg.get("maxAccel", 6.0))

print("motion profile vi, vf, max_accel:", vi, vf, max_accel)

segments = cfg["segments"]
lengths = []
for i, seg in enumerate(segments, start=1):
    s = seg["start"]
    e = seg["end"]
    b = seg["curveIntensity"]
    P0 = [s["x"], s["y"]]
    P2 = [e["x"], e["y"]]
    tg = TrajectoryGeneration(P0, P2, b)
    low, up = tg.bounds()
    L = tg.lengthAlongTheCurve(low, up, steps=10000)
    lengths.append(L)
    print(f"segment {i} length = {L}")

total_length = sum(lengths)
print("total length =", total_length)

t = _compute_global_trapezoid_time(total_length, vi, vf, max_accel)
print("computed total time:", t)

