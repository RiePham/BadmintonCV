"""
Analyze the shuttlecock trajectory from data/detections.csv to estimate hit points
and basic rally stats.

WHY find_peaks (be ready to explain this):
A shuttlecock's height over time roughly traces a series of arcs -- up after a hit,
down until the next hit. A "hit" corresponds to a local minimum (lowest point, just
before/at contact) or a local maximum (top of the arc) in the y-coordinate over time,
depending on how you define it. scipy.signal.find_peaks finds these local extrema
directly from the smoothed trajectory -- much simpler and more robust than trying to
hand-write threshold rules.

HOW TO RUN:
    python scripts/04_analyze_trajectory.py
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

df = pd.read_csv("data/detections.csv")
shuttle = df[df["class_name"] == "shuttlecock"].sort_values("frame")

if shuttle.empty:
    print("No shuttlecock detections found in data/detections.csv.")
    print("Check the class name in your dataset's data.yaml matches 'shuttlecock' exactly,")
    print("or adjust the filter above to match your dataset's actual class name.")
    exit()

# Smooth the y-trajectory to reduce detection noise before peak-finding
y = shuttle["y_center"].values
frames = shuttle["frame"].values
window = 5
y_smooth = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values

# Note: in image coordinates, y increases DOWNWARD. So a "peak" in y_smooth is the
# LOWEST point of the shuttlecock on screen (near the ground/net), and a "trough" is
# the HIGHEST point in the air (top of the arc).
peaks, _ = find_peaks(y_smooth, distance=10, prominence=5)
troughs, _ = find_peaks(-y_smooth, distance=10, prominence=5)

print(f"Detected {len(shuttle)} shuttlecock detections across {frames.max() - frames.min()} frames")
print(f"Estimated {len(peaks)} low points and {len(troughs)} high points in the trajectory")
print("These roughly correspond to hit/near-ground moments and top-of-arc moments.")

# Plot for a sanity check -- does this visually make sense against the actual video?
plt.figure(figsize=(12, 5))
plt.plot(frames, y_smooth, label="Smoothed shuttlecock height (y)")
plt.scatter(frames[peaks], y_smooth[peaks], color="red", label="Low points (near hit)", zorder=5)
plt.scatter(frames[troughs], y_smooth[troughs], color="green", label="High points (arc top)", zorder=5)
plt.gca().invert_yaxis()  # so "up" on the chart matches "up" in the video
plt.xlabel("Frame")
plt.ylabel("Y position (pixels, inverted)")
plt.title("Shuttlecock Trajectory — Estimated Hit Points")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/trajectory_analysis.png")
print("\nChart saved to outputs/trajectory_analysis.png — check it against the video.")
