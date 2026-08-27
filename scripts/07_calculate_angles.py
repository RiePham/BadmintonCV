"""
Estimate shot-contact moments from wrist SPEED (not raw elbow angle).

WHY SPEED INSTEAD OF ANGLE:
A large elbow angle just means the arm is extended -- that happens both during
an actual swing AND while a player is simply holding position, waiting for the
shuttle. What actually distinguishes a swing is FAST MOTION: the wrist stays
relatively still while the player positions, then accelerates sharply during
the swing, reaching peak speed near contact, then decelerates. So we track
wrist speed (distance moved per frame) and look for peaks in THAT signal --
this lines up much better with genuine swings than a static angle measurement.

Elbow angle is still recorded at each detected peak, since it's a useful extra
feature for the shot-classification step later (Day 3 / PyTorch).

Run: python scripts/07_calculate_angles.py
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

FPS = 30  # adjust if your video's actual frame rate is different

df = pd.read_csv("data/pose_landmarks.csv")


def get_point(frame_df, name):
    row = frame_df[frame_df["landmark_name"] == name]
    if row.empty:
        return None
    return np.array([row["x"].values[0], row["y"].values[0]])


def calculate_angle(a, b, c):
    """Angle at point b, formed by segments b->a and b->c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


frames = sorted(df["frame"].unique())
wrist_positions = []
elbow_angles = []
valid_frames = []

for f in frames:
    frame_df = df[df["frame"] == f]

    shoulder = get_point(frame_df, "right_shoulder")
    elbow = get_point(frame_df, "right_elbow")
    wrist = get_point(frame_df, "right_wrist")

    if shoulder is None or elbow is None or wrist is None:
        continue

    wrist_positions.append(wrist)
    elbow_angles.append(calculate_angle(shoulder, elbow, wrist))
    valid_frames.append(f)

print(f"Tracked wrist across {len(valid_frames)}/{len(frames)} frames")

wrist_positions = np.array(wrist_positions)  # shape (N, 2), normalized 0-1 coords
elbow_angles = np.array(elbow_angles)
valid_frames = np.array(valid_frames)

# --- Wrist speed: frame-to-frame Euclidean distance ---
# Coordinates are normalized (0-1), so this is "fraction of frame width per frame",
# not a real-world unit -- fine, since we only care about relative peaks, not
# absolute speed values.
deltas = np.diff(wrist_positions, axis=0)
speed = np.linalg.norm(deltas, axis=1)
speed = np.insert(speed, 0, 0)  # align length with valid_frames (first frame has no prior point)

# Smooth to reduce per-frame jitter from landmark detection noise
smoothed_speed = pd.Series(speed).rolling(5, center=True, min_periods=1).mean().values

# distance=15 -> don't count two peaks closer than 15 frames apart as separate swings
# prominence tuned relative to the smoothed speed's typical scale; adjust if you get
# too many/too few peaks
peaks, _ = find_peaks(smoothed_speed, distance=15, prominence=smoothed_speed.std() * 1.5)

print(f"Estimated {len(peaks)} swing moments (peak wrist speed)")

# --- Plot both signals for a sanity check ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(valid_frames, smoothed_speed, label="Wrist speed (smoothed)", color="tab:blue")
ax1.scatter(valid_frames[peaks], smoothed_speed[peaks], color="red", zorder=5, label="Estimated swing")
ax1.set_ylabel("Wrist speed (normalized units/frame)")
ax1.set_title("Wrist Speed Over Time -- Swing Detection")
ax1.legend()

ax2.plot(valid_frames, elbow_angles, label="Elbow angle", color="tab:green", alpha=0.6)
ax2.scatter(valid_frames[peaks], elbow_angles[peaks], color="red", zorder=5, label="Estimated swing")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Elbow angle (degrees)")
ax2.set_title("Elbow Angle at Estimated Swing Moments (for reference)")
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/swing_detection_analysis.png")
print("Chart saved to outputs/swing_detection_analysis.png")

# --- Save features for the next step (PyTorch) ---
features_df = pd.DataFrame({
    "frame": valid_frames[peaks],
    "approx_second": (valid_frames[peaks] / FPS).round(1),
    "wrist_speed": smoothed_speed[peaks].round(4),
    "elbow_angle": elbow_angles[peaks].round(1),
})
features_df.to_csv("data/shot_features.csv", index=False)
print(f"Saved {len(features_df)} estimated shots to data/shot_features.csv")
print("\nTip: use the 'approx_second' column to jump straight to each moment in the video.")
