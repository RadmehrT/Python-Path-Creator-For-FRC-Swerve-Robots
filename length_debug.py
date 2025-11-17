from generate_input_from_csv import build_config_from_csv
from TrajectoryGeneration import TrajectoryGeneration

cfg = build_config_from_csv("Math IA Waypoints list - Sheet1.csv")
steps = cfg.get("steps", 100)
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

print("total length =", sum(lengths))

