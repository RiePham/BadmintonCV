"""
================================================================================
FILE 10: run_on_video.py  (the "put it all together" demo file)
================================================================================

HOW TO RUN THIS FILE:
    1. Open this file and change INPUT_VIDEO below to your video's path
    2. python scripts/10_run_on_video.py

WHAT THIS FILE DOES (super simple explanation):
    This file is like a factory assembly line: it takes ANY raw badminton
    video and automatically runs EVERY earlier step in order, without you
    having to run each file by hand:
        Step 1: find the body joints in every frame       (like file 06)
        Step 2: guess when swings/hits happened            (like file 08)
        Step 3: ask the trained robot (file 09) "what shot was this?"
        Step 4: draw the guessed shot name on the video at that moment
    At the end, you get a brand new video where you can literally SEE the
    robot's guesses appear on screen, like subtitles.

    HONEST NOTE: this reads the WHOLE video first, then makes a NEW video
    -- it does NOT work on a live camera in real time. Also, the robot in
    file 09 only ever studied ONE video with 33 examples, so if you run
    this on a DIFFERENT video, don't expect the guesses to be very
    accurate -- this file is a demo of the PIPELINE working end-to-end,
    not proof of a highly accurate model.

WHAT GOES IN, WHAT COMES OUT:
    IN:  videos/testvid2.mp4 (or whatever video you set below)
    IN:  outputs/shot_classifier.pt   <- the trained robot brain from file 09
    IN:  data/shot_features_v2.csv    <- used to remember label names/scaling from file 09
    OUT: outputs/testvid2_annotated.mp4  <- new video with guesses drawn on it

HOW THIS FILE CONNECTS TO OTHER FILES:
    This file basically GLUES TOGETHER files 06 + 08 + 09 into one chain,
    so you don't have to run them one by one:

    (raw video)
         |
         v
    [Step 1 -- same idea as 06_pose_extraction.py]
         |
         v
    [Step 2 -- same idea as 08_calculate_contacts_v2.py]
         |
         v
    [Step 3 -- loads the trained brain from 09_train_shot_classifier.py]
         |
         v
    [Step 4 -- draws the answer onto the video]
         |
         v
    outputs/testvid2_annotated.mp4  (final result you can watch)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
import torch
import torch.nn as nn
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------------------------------------------------------------
# CONFIG -- EDIT THESE before running
# ------------------------------------------------------------------------
INPUT_VIDEO = "videos/testvid2.mp4"          # <-- set this to the real path
OUTPUT_VIDEO = "outputs/testvid2_annotated.mp4"
MODEL_WEIGHTS = "outputs/shot_classifier.pt"
TRAIN_CSV = "data/shot_features_v2.csv"       # used to refit the label encoder + scaler
                                               # (must match what 09_train_shot_classifier.py used)
FPS_OVERRIDE = None  # set a number if you know the video's fps and want to force it

# These next settings must match file 08's settings, since Step 2 below
# reuses the exact same "still -> burst" swing-finding logic.
MIN_VISIBILITY = 0.5
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2
MIN_PEAK_SEPARATION_FRAMES = 15
SPEED_FLOOR_PERCENTILE = 70
MAX_WINDUP_FRAMES = 20
BURST_RATE_PERCENTILE = 40
STILLNESS_SMOOTH_FRAMES = 3
MAX_RELIABLE_FRAME_GAP = 1

LABEL_DISPLAY_SECONDS = 0.6   # how long each predicted label stays on screen

POSE_MODEL_PATH = "pose_landmarker.task"  # same model file used by 06_pose_extraction.py

# Same 33 body-joint names used in file 06, in the same order
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


# ------------------------------------------------------------------------
# Model definition -- MUST match 09_train_shot_classifier.py exactly
# ------------------------------------------------------------------------
class ShotClassifier(nn.Module):
    """
    SIMPLE EXPLANATION:
    This must be built EXACTLY the same shape as the robot brain in file
    09 -- otherwise the saved weights (numbers) won't fit back into it
    correctly, like trying to put a puzzle piece into the wrong puzzle.
    """
    def __init__(self, n_features, n_classes, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------------
# Step 1: pose extraction  (same idea as file 06)
# ------------------------------------------------------------------------
def extract_pose_landmarks(video_path):
    """
    SIMPLE EXPLANATION:
    Watches the WHOLE input video and, for every frame, records where all
    33 body joints are. Returns this as one big table (like the CSV file
    06 makes), but kept in memory instead of saved to disk, since we're
    going to use it right away in the next steps.
    """
    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at '{POSE_MODEL_PATH}'. This must be the same "
            f"pose_landmarker.task file used by scripts/06_pose_extraction.py, "
            f"kept in the repo root (or update POSE_MODEL_PATH above)."
        )

    base_options = mp_python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = FPS_OVERRIDE or cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / fps))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                for i, lm in enumerate(landmarks):
                    rows.append({
                        "frame": frame_idx,
                        "landmark_id": i,
                        "landmark_name": LANDMARK_NAMES[i],
                        "x": lm.x,
                        "y": lm.y,
                        "visibility": lm.visibility,
                    })
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  ...processed {frame_idx} frames")

    cap.release()
    print(f"Extracted pose for {frame_idx} frames from {video_path} (fps={fps:.2f})")
    return pd.DataFrame(rows), fps, frame_w, frame_h


# ------------------------------------------------------------------------
# Step 2: contact detection (same idea as file 08)
# ------------------------------------------------------------------------
def get_point(frame_df, name, min_visibility=MIN_VISIBILITY):
    """SIMPLE EXPLANATION: same as file 08 -- find one joint's (x, y) spot
    in one frame, or None if MediaPipe wasn't confident about it."""
    row = frame_df[frame_df["landmark_name"] == name]
    if row.empty:
        return None
    vis = row["visibility"].values[0]
    if pd.notna(vis) and vis < min_visibility:
        return None
    return np.array([row["x"].values[0], row["y"].values[0]])


def calculate_angle(a, b, c):
    """SIMPLE EXPLANATION: same as file 08 -- measures the elbow angle
    from the shoulder(a), elbow(b), wrist(c) points."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def smooth_track(positions):
    """SIMPLE EXPLANATION: same as file 08 -- smooths a wobbly (x, y) path
    into a cleaner curve."""
    n = len(positions)
    if n < 2:
        return positions.copy()
    window = SAVGOL_WINDOW
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < SAVGOL_POLYORDER + 2:
        return positions.copy()
    return np.column_stack([
        savgol_filter(positions[:, 0], window, SAVGOL_POLYORDER),
        savgol_filter(positions[:, 1], window, SAVGOL_POLYORDER),
    ])


def smooth_track_by_segment(positions, frame_gap):
    """SIMPLE EXPLANATION: same as file 08 -- smooths in separate chunks so
    a gap of dropped frames never gets blended across."""
    n = len(positions)
    out = np.empty_like(positions, dtype=float)
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or frame_gap[i] > 1:
            out[seg_start:i] = smooth_track(positions[seg_start:i])
            seg_start = i
    return out


def load_pose_signals(df):
    """SIMPLE EXPLANATION: same idea as file 08's load_pose_signals, but
    reads from the in-memory table made by Step 1 above, instead of a
    saved CSV file, since this whole pipeline runs in one go."""
    frames = sorted(df["frame"].unique())
    raw_shoulder, raw_elbow, raw_wrist, valid_frames = [], [], [], []
    for f in frames:
        frame_df = df[df["frame"] == f]
        shoulder = get_point(frame_df, "right_shoulder")
        elbow = get_point(frame_df, "right_elbow")
        wrist = get_point(frame_df, "right_wrist")
        if shoulder is None or elbow is None or wrist is None:
            continue
        raw_shoulder.append(shoulder)
        raw_elbow.append(elbow)
        raw_wrist.append(wrist)
        valid_frames.append(f)

    if len(valid_frames) < 5:
        raise RuntimeError("Too few valid pose frames -- check the video / MediaPipe detection.")

    valid_frames = np.array(valid_frames)
    raw_shoulder, raw_elbow, raw_wrist = np.array(raw_shoulder), np.array(raw_elbow), np.array(raw_wrist)

    frame_gap = np.diff(valid_frames, prepend=valid_frames[0])
    frame_gap[0] = 1

    smooth_shoulder = smooth_track_by_segment(raw_shoulder, frame_gap)
    smooth_elbow = smooth_track_by_segment(raw_elbow, frame_gap)
    smooth_wrist = smooth_track_by_segment(raw_wrist, frame_gap)

    elbow_angles = np.array([
        calculate_angle(smooth_shoulder[i], smooth_elbow[i], smooth_wrist[i])
        for i in range(len(valid_frames))
    ])

    deltas = np.diff(smooth_wrist, axis=0)
    dist = np.insert(np.linalg.norm(deltas, axis=1), 0, 0.0)
    speed = dist / frame_gap

    reliable = frame_gap <= MAX_RELIABLE_FRAME_GAP
    reliable[0] = False

    return valid_frames, speed, elbow_angles, reliable


def find_swing_candidates(speed, reliable):
    """SIMPLE EXPLANATION: same "still -> burst" swing-finding logic as
    file 08 -- look for sharp speed bumps that rise fast out of stillness."""
    speed_for_peaks = np.where(reliable, speed, 0.0)
    reliable_speed = speed[reliable]
    if len(reliable_speed) == 0:
        return []

    floor = np.percentile(reliable_speed, SPEED_FLOOR_PERCENTILE)
    peaks, _ = find_peaks(
        speed_for_peaks,
        distance=MIN_PEAK_SEPARATION_FRAMES,
        prominence=reliable_speed.std() * 1.0,
        height=floor,
    )

    gap_positions = np.where(~reliable)[0]
    candidates = []
    for p in peaks:
        earlier_gaps = gap_positions[gap_positions <= p]
        gap_floor = int(earlier_gaps[-1]) if len(earlier_gaps) else 0
        lookback_start = max(0, p - MAX_WINDUP_FRAMES, gap_floor)
        window = speed[lookback_start:p + 1]
        if len(window) < 2:
            continue

        smoothed_window = pd.Series(window).rolling(
            STILLNESS_SMOOTH_FRAMES, min_periods=1, center=True
        ).mean().values
        local_min_offset = int(np.argmin(smoothed_window))
        still_idx = lookback_start + local_min_offset
        windup_frames = p - still_idx
        if windup_frames < 1:
            continue

        candidates.append({"peak_idx": p, "windup_frames": windup_frames})

    return candidates


# ------------------------------------------------------------------------
# Step 3: classify each candidate  (uses the robot brain from file 09)
# ------------------------------------------------------------------------
def load_classifier_and_preprocessing():
    """
    SIMPLE EXPLANATION:
    Loads the trained robot brain (file 09's saved weights) back into
    memory, and also rebuilds the exact same "number scaling" and
    "label name" setup that file 09 used -- this MUST match exactly, or
    the robot's guesses would come out scrambled/wrong.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["shot_type"] = train_df["shot_type"].astype(str).str.strip().str.lower()
    train_df = train_df[train_df["shot_type"].notna() & (train_df["shot_type"] != "") & (train_df["shot_type"] != "nan")]
    train_df = train_df.dropna(subset=["wrist_speed", "windup_frames", "elbow_angle"])

    feature_cols = ["wrist_speed", "windup_frames", "elbow_angle"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_raw = train_df["shot_type"].values

    encoder = LabelEncoder()
    encoder.fit(y_raw)

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = ShotClassifier(n_features=len(feature_cols), n_classes=len(encoder.classes_))
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()  # tell the model "we're just asking questions now, not learning"

    return model, scaler, encoder, feature_cols


def classify_candidates(candidates, valid_frames, speed, elbow_angles, model, scaler, encoder, feature_cols):
    """
    SIMPLE EXPLANATION:
    For each candidate swing moment found in Step 2, grab its 3 clue
    numbers, hand them to the trained robot, and record its best guess
    plus how confident it was (as a percentage).
    """
    results = []
    for c in candidates:
        i = c["peak_idx"]
        row = pd.DataFrame([{
            "wrist_speed": speed[i],
            "windup_frames": c["windup_frames"],
            "elbow_angle": elbow_angles[i],
        }])
        X = scaler.transform(row[feature_cols].values.astype(np.float32))
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)  # turn scores into percentages that add up to 100%
            pred_idx = probs.argmax(dim=1).item()  # pick the highest-scoring answer
            confidence = probs[0, pred_idx].item()
        label = encoder.inverse_transform([pred_idx])[0]  # turn the number answer back into text
        results.append({
            "frame": int(valid_frames[i]),
            "label": label,
            "confidence": confidence,
        })
    return results


# ------------------------------------------------------------------------
# Step 4: render annotated video
# ------------------------------------------------------------------------
def render_annotated_video(input_video, output_video, predictions, fps):
    """
    SIMPLE EXPLANATION:
    Plays through the video ONE more time, and this time, whenever we're
    near a frame the robot made a guess about, we draw text on screen
    (like "SMASH (82%)") for a short moment, then save the whole thing as
    a brand new video file.
    """
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    display_frames = int(LABEL_DISPLAY_SECONDS * fps)
    # map: frame_idx -> (label, confidence) for every frame it should still be shown on
    label_by_frame = {}
    for pred in predictions:
        for f in range(pred["frame"], pred["frame"] + display_frames):
            label_by_frame[f] = pred

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in label_by_frame:
            pred = label_by_frame[frame_idx]
            text = f"{pred['label'].upper()} ({pred['confidence']*100:.0f}%)"
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved annotated video to {output_video}")


# ------------------------------------------------------------------------
# Main -- runs Steps 1 -> 2 -> 3 -> 4 in order
# ------------------------------------------------------------------------
def main():
    """
    SIMPLE EXPLANATION: this is the "assembly line manager" -- it just
    calls each step, in order, printing progress along the way.
    """
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(
            f"INPUT_VIDEO not found: {INPUT_VIDEO}\n"
            f"Edit the INPUT_VIDEO path at the top of this script."
        )
    os.makedirs("outputs", exist_ok=True)

    print("Step 1/4: extracting pose landmarks...")
    pose_df, fps, w, h = extract_pose_landmarks(INPUT_VIDEO)

    print("Step 2/4: detecting contact candidates...")
    valid_frames, speed, elbow_angles, reliable = load_pose_signals(pose_df)
    candidates = find_swing_candidates(speed, reliable)
    print(f"  Found {len(candidates)} candidate contact moments")

    print("Step 3/4: classifying each candidate...")
    model, scaler, encoder, feature_cols = load_classifier_and_preprocessing()
    predictions = classify_candidates(candidates, valid_frames, speed, elbow_angles,
                                       model, scaler, encoder, feature_cols)
    for p in predictions:
        print(f"  frame {p['frame']:5d} ({p['frame']/fps:5.2f}s): "
              f"{p['label']:10s} (confidence {p['confidence']*100:.0f}%)")

    print("Step 4/4: rendering annotated video...")
    render_annotated_video(INPUT_VIDEO, OUTPUT_VIDEO, predictions, fps)


if __name__ == "__main__":
    main()