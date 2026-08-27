"""
================================================================================
FILE 07: calculate_angles.py   (this is "v1" -- the FIRST, IMPERFECT attempt)
================================================================================

HOW TO RUN THIS FILE:
    python scripts/07_calculate_angles.py

WHAT THIS FILE DOES (super simple explanation):
    This file tries to guess WHEN in the video the player actually hit the
    shuttlecock, just by looking at how fast their wrist was moving.
    Idea: the wrist moves SLOW while just standing/waiting, then moves
    FAST right when swinging the racket, then slows down again. So we
    measure "how far did the wrist move between this frame and the last
    frame" (that's "speed"), and look for the bumps (peaks) in that speed.

    IMPORTANT: this file has KNOWN PROBLEMS (explained in file 08's
    comments) -- it sometimes thinks a swing happened when really it's
    just noise or a mistake in tracking. That's WHY file 08 exists: file
    08 is the fixed, better version. This file (07) is kept only so you
    can show/explain the "before" version if asked in an interview.

WHAT GOES IN, WHAT COMES OUT:
    IN:  data/pose_landmarks.csv     <- made by 06_pose_extraction.py
    OUT: outputs/swing_detection_analysis.png  <- a chart
    OUT: data/shot_features.csv      <- list of guessed swing moments (v1, old)

HOW THIS FILE CONNECTS TO OTHER FILES:
    06_pose_extraction.py
              |
              v
    data/pose_landmarks.csv
              |
              v
    07_calculate_angles.py   <-- YOU ARE HERE (the OLD, replaced version)
              |
              v
    data/shot_features.csv   <-- NOT used by any later file anymore!
                                  (08_calculate_contacts_v2.py replaced this
                                  entirely with a better approach and writes
                                  to a DIFFERENT file: shot_features_v2.csv)
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

FPS = 30  # adjust if your video's actual frame rate is different

# Load the joint-position spreadsheet made by file 06
df = pd.read_csv("data/pose_landmarks.csv")


def get_point(frame_df, name):
    """
    SIMPLE EXPLANATION:
    Given all the joint rows for ONE frame, find the row for a specific
    joint (like "right_wrist") and return its (x, y) position as a pair
    of numbers. Returns None if that joint wasn't found in this frame.
    """
    row = frame_df[frame_df["landmark_name"] == name]
    if row.empty:
        return None
    return np.array([row["x"].values[0], row["y"].values[0]])


def calculate_angle(a, b, c):
    """
    SIMPLE EXPLANATION:
    Given 3 points (shoulder, elbow, wrist), this measures the angle AT
    the elbow -- like measuring how "bent" or "straight" the arm is,
    using basic geometry/trigonometry (the dot product formula).
    A straight arm = angle near 180 degrees. A tightly bent arm = a small
    angle number.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ------------------------------------------------------------------------
# STEP 1: go through every frame and grab the shoulder/elbow/wrist points
# ------------------------------------------------------------------------
frames = sorted(df["frame"].unique())
wrist_positions = []
elbow_angles = []
valid_frames = []

for f in frames:
    frame_df = df[df["frame"] == f]

    shoulder = get_point(frame_df, "right_shoulder")
    elbow = get_point(frame_df, "right_elbow")
    wrist = get_point(frame_df, "right_wrist")

    # Skip this frame entirely if MediaPipe couldn't find all 3 points
    if shoulder is None or elbow is None or wrist is None:
        continue

    wrist_positions.append(wrist)
    elbow_angles.append(calculate_angle(shoulder, elbow, wrist))
    valid_frames.append(f)

print(f"Tracked wrist across {len(valid_frames)}/{len(frames)} frames")

wrist_positions = np.array(wrist_positions)  # shape (N, 2), normalized 0-1 coords
elbow_angles = np.array(elbow_angles)
valid_frames = np.array(valid_frames)

# ------------------------------------------------------------------------
# STEP 2: measure how fast the wrist is moving, frame to frame
# ------------------------------------------------------------------------
# "Speed" here just means: how far did the wrist move compared to the
# PREVIOUS frame? (straight-line distance between the two points)
deltas = np.diff(wrist_positions, axis=0)
speed = np.linalg.norm(deltas, axis=1)
speed = np.insert(speed, 0, 0)  # first frame has no "previous" frame, so speed = 0 there

# Smooth the speed a little bit to reduce jitter (tiny detection mistakes)
smoothed_speed = pd.Series(speed).rolling(5, center=True, min_periods=1).mean().values

# ------------------------------------------------------------------------
# STEP 3: find the bumps (peaks) in the speed -- these are our GUESSED swings
# ------------------------------------------------------------------------
# distance=15 -> don't count two peaks closer than 15 frames apart as separate swings
# prominence tuned relative to the smoothed speed's typical scale; adjust if you get
# too many/too few peaks
peaks, _ = find_peaks(smoothed_speed, distance=15, prominence=smoothed_speed.std() * 1.5)

print(f"Estimated {len(peaks)} swing moments (peak wrist speed)")

# ------------------------------------------------------------------------
# STEP 4: draw a chart so we can visually sanity-check the guesses
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# STEP 5: save the guessed swing moments to a spreadsheet
# ------------------------------------------------------------------------
features_df = pd.DataFrame({
    "frame": valid_frames[peaks],
    "approx_second": (valid_frames[peaks] / FPS).round(1),
    "wrist_speed": smoothed_speed[peaks].round(4),
    "elbow_angle": elbow_angles[peaks].round(1),
})
features_df.to_csv("data/shot_features.csv", index=False)
print(f"Saved {len(features_df)} estimated shots to data/shot_features.csv")
print("\nTip: use the 'approx_second' column to jump straight to each moment in the video.")