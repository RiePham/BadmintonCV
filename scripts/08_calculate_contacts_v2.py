"""
V2 - Estimate shot-CONTACT moments (not just "swings") from pose + (optional) audio.

WHY V1 (07_calculate_angles.py) WAS GIVING FALSE HITS
Wrist speed alone is ambiguous for a few reasons that all showed up as your
"detected hit before actual contact" problem:

  1. NOISY LANDMARKS DURING FAST MOTION. MediaPipe struggles most exactly when
     the racket arm is moving fastest -- motion blur + the racket itself
     occluding the wrist. A couple of frames where the wrist position is
     briefly guessed wrong looks EXACTLY like a huge speed spike, even with
     smoothing, because smoothing a rolling mean over noisy data still lets
     single bad frames distort the result.
  2. ONE GLOBAL THRESHOLD FOR THE WHOLE VIDEO. A smash and a soft net shot
     have very different wrist speeds. A single `std() * 1.5` prominence
     either misses soft shots or fires on noise during fast rallies.
     NOTE: this version still uses percentile/std-based thresholds (see
     SPEED_FLOOR_PERCENTILE, BURST_RATE_PERCENTILE below) -- that part of
     the problem is narrowed (two independent signals must agree instead of
     one), not eliminated. A truly adaptive per-shot-type threshold would
     need shot-type labels or the shuttle-trajectory fusion described at
     the bottom of this file. Don't oversell this as "fixed."
  3. NO "WAS THE ARM ACTUALLY STILL FIRST?" CHECK. Your own read of the
     footage -- players keep the racket fairly still while moving, and only
     swing right before contact -- is exactly the right signal to use
     directly, and V1 wasn't using it. A real swing is a short, sharp BURST
     out of near-stillness, not just "the fastest point in some window."

WHAT THIS VERSION ADDS
  A. Filter out low-confidence landmarks (MediaPipe's `visibility` score) so
     occlusion/motion-blur guesses don't get treated as real movement.
  B. Smooth the wrist TRAJECTORY itself (Savitzky-Golay), not the speed
     signal after the fact -- this keeps peak timing much more accurate.
  C. Require a genuine "still -> burst" pattern: for every candidate speed
     peak, look backward for the nearest local stillness point, and only
     keep the candidate if the rise from still->peak happened fast (few
     frames) and hard (steep). Slow, gradual speed-ups (repositioning,
     footwork) get rejected even if they eventually reach a locally-highest
     speed value. The "stillness" reference itself is a short rolling
     average (not a single raw frame), so one noisy low reading can't fake
     a "still" point the way one noisy high reading fakes a swing peak --
     the same anti-noise principle from (A)/(B), applied here too.
  D. GAP-AWARE speed calculation (the fix that actually mattered most once
     this was tested against real data -- see below). Filtering out
     low-visibility frames in (A) means some frames get dropped entirely,
     so the frames that survive are NOT evenly spaced 1-apart anymore.
     Diffing wrist position across surviving samples while ignoring that
     gap turns "19 real frames were dropped here" into "the wrist moved
     this whole distance in 1 frame" -- a fake spike far bigger than any
     real swing, and it happens most often exactly where you'd expect real
     swings (visibility drops during motion blur, i.e. near real contacts).
     Checked directly against data/pose_landmarks.csv: 12 of the 15 highest
     "speed" frames produced by the naive diff were gap artifacts, not real
     motion; the single highest "speed" moment in the whole clip was a
     19-frame gap collapsed into one step. This version normalizes each
     speed sample by the actual number of elapsed frames, and only trusts
     a sample as a real instantaneous reading (usable as a peak, or as a
     "stillness" reference point) when it spans an exact, unbroken
     consecutive frame pair. In this dataset that's 917/939 samples
     (97.7%) -- excluding the rest costs almost nothing and removes the
     fake-spike risk entirely.
  E. (Optional, biggest accuracy win) Fuse with AUDIO onset detection. The
     "pok" of shuttle contact is short and acoustically distinct from
     footsteps/crowd noise. Published racket-sport hit-detection work
     (tennis / table tennis) reports 85-95%+ accuracy from audio onsets,
     but that's from purpose-built classifiers (MFCC/spectral features +
     a trained model), not raw generic onset detection. This script uses
     librosa's generic `onset_detect` as a first pass -- treat its matches
     as a useful extra signal, not a ground-truth oracle, especially with
     crowd noise, umpire calls, or racket-on-racket sounds from the other
     player. If a pose-based candidate has a matching audio onset nearby,
     we snap the contact frame to the audio onset (more precise) and mark
     it confirmed. If a candidate has NO nearby audio onset, it's flagged
     as suspicious rather than silently kept or dropped -- you decide the
     cutoff.

WHAT THIS VERSION STILL DOESN'T DO (documented for later, not needed yet)
  The highest-accuracy published approach (e.g. the TrackNet-shuttle +
  YOLO-swing fusion paper, 89.7% accuracy / 91.3% recall vs 58.8% accuracy
  from shuttle-trajectory alone) watches the SHUTTLE and flags a hit when
  its flight direction reverses sharply near a player. That needs a
  shuttle detector/tracker, which is a separate model you don't have yet
  at this stage of the project. Once you build one, add a step here: for
  each surviving candidate, check whether the shuttle is close to the
  wrist AND reverses direction within ~3 frames. That's a strict superset
  of what's below, not a replacement -- keep this script's logic either way.

REQUIREMENTS
  pip install librosa   (only needed if AUDIO_PATH / VIDEO_PATH is set)
  ffmpeg must be installed and on PATH (only needed to auto-extract audio
  from a video file)

Run: python scripts/08_calculate_contacts_v2.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# CONFIG -- tweak these for your footage
# ------------------------------------------------------------------------
FPS = 30  # your badminton clip is 25fps -- keep this in sync with the source video
LANDMARKS_CSV = "data/pose_landmarks.csv"

# Set to a video file path to auto-extract audio and fuse it in.
# Leave as None to skip audio fusion (script still works, just pose-only).
VIDEO_PATH = None  # e.g. "videos/videomomota.mp4"
AUDIO_MATCH_TOLERANCE_FRAMES = 5  # how close an audio onset must be (~200ms @25fps)

MIN_VISIBILITY = 0.5           # drop landmark readings MediaPipe wasn't confident about
SAVGOL_WINDOW = 7              # must be odd; smaller = less smoothing
SAVGOL_POLYORDER = 2
MIN_PEAK_SEPARATION_FRAMES = 15
SPEED_FLOOR_PERCENTILE = 70    # ignore speed peaks below this percentile (relative, not a magic number)
MAX_WINDUP_FRAMES = 20         # a real swing's still->burst rise shouldn't take longer than this (~0.8s @25fps)
BURST_RATE_PERCENTILE = 40     # candidates with a weaker still->peak rise than this percentile get flagged
STILLNESS_SMOOTH_FRAMES = 3    # "stillness" reference = a short rolling average, not one raw frame
MAX_RELIABLE_FRAME_GAP = 1     # a speed sample is only trusted as instantaneous if it spans exactly
                                # this many real frames (1 = truly consecutive, no dropped frames between)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------------------
# Pose loading helpers
# ------------------------------------------------------------------------
def get_point(frame_df, name, min_visibility=MIN_VISIBILITY):
    row = frame_df[frame_df["landmark_name"] == name]
    if row.empty:
        return None
    if "visibility" in row.columns:
        vis = row["visibility"].values[0]
        if pd.notna(vis) and vis < min_visibility:
            return None  # MediaPipe itself wasn't confident -- don't trust this frame
    return np.array([row["x"].values[0], row["y"].values[0]])


def calculate_angle(a, b, c):
    """Angle at point b, formed by segments b->a and b->c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def smooth_track(positions):
    """Savitzky-Golay smoothing of an (N, 2) trajectory. Falls back to raw
    data if there aren't enough points for the configured window."""
    n = len(positions)
    if n < 2:
        return positions.copy()
    window = SAVGOL_WINDOW
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < SAVGOL_POLYORDER + 2:
        return positions.copy()  # too short a run to smooth meaningfully
    smoothed = np.column_stack([
        savgol_filter(positions[:, 0], window, SAVGOL_POLYORDER),
        savgol_filter(positions[:, 1], window, SAVGOL_POLYORDER),
    ])
    return smoothed


def smooth_track_by_segment(positions, frame_gap):
    """Savitzky-Golay assumes evenly-spaced samples. Once low-visibility
    frames are dropped, the surviving samples are NOT evenly spaced across a
    gap -- so smoothing straight through a gap silently blends real motion
    with however many frames were skipped. Instead, split into contiguous
    runs (frame_gap == 1 throughout) and smooth each run independently.

    `frame_gap[i]` = valid_frames[i] - valid_frames[i-1] (frame_gap[0] is
    unused / set to 1 by the caller)."""
    n = len(positions)
    out = np.empty_like(positions, dtype=float)
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or frame_gap[i] > 1:
            out[seg_start:i] = smooth_track(positions[seg_start:i])
            seg_start = i
    return out


def load_pose_signals(csv_path):
    df = pd.read_csv(csv_path)
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

    print(f"Tracked wrist across {len(valid_frames)}/{len(frames)} frames "
          f"(rest dropped: missing or low-visibility landmarks)")

    valid_frames = np.array(valid_frames)
    raw_shoulder = np.array(raw_shoulder)
    raw_elbow = np.array(raw_elbow)
    raw_wrist = np.array(raw_wrist)

    # How many real frames separate each surviving sample from the previous
    # one. frame_gap[0] is meaningless (no "previous" sample) -- set to 1 so
    # it's treated like a normal step and never flagged as a gap.
    frame_gap = np.diff(valid_frames, prepend=valid_frames[0])
    frame_gap[0] = 1
    n_gaps = int((frame_gap > MAX_RELIABLE_FRAME_GAP).sum())
    if n_gaps:
        print(f"NOTE: {n_gaps} frame gap(s) from dropped low-visibility frames "
              f"(largest: {frame_gap.max()} frames) -- those spans are excluded "
              f"from swing detection, not trusted as instantaneous motion")

    # Smooth trajectories BEFORE differentiating -- this is the key fix vs V1,
    # which smoothed the (already noisy-derivative) speed signal after the
    # fact. Smoothed per contiguous run (see smooth_track_by_segment) so a
    # gap of dropped frames never gets blended across.
    smooth_shoulder = smooth_track_by_segment(raw_shoulder, frame_gap)
    smooth_elbow = smooth_track_by_segment(raw_elbow, frame_gap)
    smooth_wrist = smooth_track_by_segment(raw_wrist, frame_gap)

    elbow_angles = np.array([
        calculate_angle(smooth_shoulder[i], smooth_elbow[i], smooth_wrist[i])
        for i in range(len(valid_frames))
    ])

    deltas = np.diff(smooth_wrist, axis=0)
    dist = np.linalg.norm(deltas, axis=1)
    dist = np.insert(dist, 0, 0.0)
    # Normalize by how many real frames each step actually spans, so a step
    # across a dropped-frame gap reads as "average speed over that span"
    # instead of "all of that distance happened in one frame."
    speed = dist / frame_gap

    accel = np.diff(speed)
    accel = np.insert(accel, 0, 0)

    reliable = frame_gap <= MAX_RELIABLE_FRAME_GAP
    reliable[0] = False  # index 0 has no real prior sample to diff against

    return valid_frames, speed, accel, elbow_angles, reliable


# ------------------------------------------------------------------------
# Core "still -> burst" candidate detection
# ------------------------------------------------------------------------
def find_swing_candidates(speed, reliable):
    # Only genuinely consecutive-frame samples may act as a swing peak --
    # an averaged-across-a-gap sample is not a real instantaneous reading.
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

    # Position of every unreliable (gap-spanning) sample, used to stop the
    # windup lookback from reading across a gap -- we have no idea what the
    # wrist actually did during a dropped-frame span, so don't guess.
    gap_positions = np.where(~reliable)[0]

    candidates = []
    for p in peaks:
        earlier_gaps = gap_positions[gap_positions <= p]
        gap_floor = int(earlier_gaps[-1]) if len(earlier_gaps) else 0
        lookback_start = max(0, p - MAX_WINDUP_FRAMES, gap_floor)
        window = speed[lookback_start:p + 1]
        if len(window) < 2:
            continue

        # "Stillness" reference = the lowest point of a short rolling
        # average, not the single lowest raw frame -- a lone noisy
        # near-zero reading shouldn't be able to fake a genuine pause any
        # more than a lone noisy high reading should fake a swing peak.
        smoothed_window = pd.Series(window).rolling(
            STILLNESS_SMOOTH_FRAMES, min_periods=1, center=True
        ).mean().values
        local_min_offset = int(np.argmin(smoothed_window))
        still_idx = lookback_start + local_min_offset
        windup_frames = p - still_idx
        if windup_frames < 1:
            continue  # peak IS the stillness point, no real windup captured

        still_speed = float(smoothed_window[local_min_offset])
        burst_rate = (speed[p] - still_speed) / windup_frames
        candidates.append({
            "peak_idx": p,
            "still_idx": still_idx,
            "windup_frames": windup_frames,
            "burst_rate": burst_rate,
        })

    if not candidates:
        return []

    rates = np.array([c["burst_rate"] for c in candidates])
    rate_floor = np.percentile(rates, BURST_RATE_PERCENTILE)
    for c in candidates:
        c["sharp_burst"] = c["burst_rate"] >= rate_floor

    return candidates


# ------------------------------------------------------------------------
# Optional audio fusion
# ------------------------------------------------------------------------
def extract_audio(video_path, out_wav="data/_extracted_audio.wav"):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "22050", out_wav],
        check=True, capture_output=True,
    )
    return out_wav


def get_audio_onset_frames(video_path, fps):
    try:
        import librosa
    except ImportError:
        print("librosa not installed (pip install librosa) -- skipping audio fusion, "
              "using pose signal only.")
        return None

    # Sanity-check the configured FPS against the actual video -- a mismatch
    # here silently shifts every audio-matched timestamp.
    import cv2
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if actual_fps and abs(actual_fps - fps) > 0.5:
        print(f"WARNING: configured FPS={fps} but {video_path} reports "
              f"{actual_fps:.2f}fps -- update the FPS constant or audio "
              f"onsets will be matched to the wrong frames.")

    try:
        wav_path = extract_audio(video_path)
    except Exception as e:
        print(f"Could not extract audio from {video_path} ({e}) -- skipping audio fusion.")
        return None

    y, sr = librosa.load(wav_path, sr=None)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    onset_frames = np.round(onset_times * fps).astype(int)
    print(f"Found {len(onset_frames)} audio onsets (candidate contact sounds)")
    return onset_frames


def match_to_audio(candidate_frame_numbers, audio_onset_frames, tolerance):
    """For each pose candidate, find the nearest audio onset within tolerance.
    Returns (matched_frame_or_None) per candidate, using the AUDIO frame when
    matched since audio timing is more precise than 30fps video sampling."""
    if audio_onset_frames is None or len(audio_onset_frames) == 0:
        return [None] * len(candidate_frame_numbers)

    matches = []
    for cf in candidate_frame_numbers:
        diffs = np.abs(audio_onset_frames - cf)
        best = np.argmin(diffs)
        matches.append(int(audio_onset_frames[best]) if diffs[best] <= tolerance else None)
    return matches


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    valid_frames, speed, accel, elbow_angles, reliable = load_pose_signals(LANDMARKS_CSV)
    candidates = find_swing_candidates(speed, reliable)
    print(f"Pose signal found {len(candidates)} still->burst candidates "
          f"({sum(c['sharp_burst'] for c in candidates)} with a sharp burst)")

    candidate_frame_numbers = [valid_frames[c["peak_idx"]] for c in candidates]

    audio_onset_frames = None
    if VIDEO_PATH:
        audio_onset_frames = get_audio_onset_frames(VIDEO_PATH, FPS)
    matched_audio = match_to_audio(candidate_frame_numbers, audio_onset_frames,
                                    AUDIO_MATCH_TOLERANCE_FRAMES)

    rows = []
    for c, cf, audio_frame in zip(candidates, candidate_frame_numbers, matched_audio):
        final_frame = audio_frame if audio_frame is not None else int(cf)
        if audio_frame is not None:
            source = "pose+audio"
        elif c["sharp_burst"]:
            source = "pose_sharp"       # trust this even without audio
        else:
            source = "pose_weak"        # keep for review, but flagged as less certain
        rows.append({
            "frame": final_frame,
            "approx_second": round(final_frame / FPS, 2),
            "wrist_speed": round(float(speed[c["peak_idx"]]), 4),
            "windup_frames": c["windup_frames"],
            "elbow_angle": round(float(elbow_angles[c["peak_idx"]]), 1),
            "source": source,
        })

    # Any audio onsets with NO matching pose candidate nearby -- these are shots
    # the pose signal may have missed entirely (e.g. a very quick net shot).
    if audio_onset_frames is not None:
        matched_audio_frames = {m for m in matched_audio if m is not None}
        for af in audio_onset_frames:
            if af in matched_audio_frames:
                continue
            if any(abs(af - r["frame"]) <= AUDIO_MATCH_TOLERANCE_FRAMES for r in rows):
                continue
            rows.append({
                "frame": int(af),
                "approx_second": round(af / FPS, 2),
                "wrist_speed": None,
                "windup_frames": None,
                "elbow_angle": None,
                "source": "audio_only",
            })

    features_df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    features_df.to_csv("data/shot_features_v2.csv", index=False)

    print("\nBreakdown by source (use this to decide what to trust):")
    print(features_df["source"].value_counts().to_string())
    print(f"\nSaved {len(features_df)} candidate contacts to data/shot_features_v2.csv")
    print("Tip: use 'approx_second' to jump to each moment and manually confirm "
          "a sample of each 'source' category before trusting it as ground truth.")

    # --- Sanity-check plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(valid_frames, speed, label="Wrist speed (Savgol-smoothed trajectory, gap-normalized)", color="tab:blue")
    ax1.scatter(valid_frames[~reliable], speed[~reliable], color="black", marker="x",
                zorder=4, label="gap-spanning (excluded from detection)", s=30)
    colors = {"pose+audio": "red", "pose_sharp": "orange", "pose_weak": "gray", "audio_only": "purple"}
    for src, color in colors.items():
        sub = features_df[features_df["source"] == src]
        if sub.empty:
            continue
        idx_in_signal = np.searchsorted(valid_frames, sub["frame"].values)
        idx_in_signal = np.clip(idx_in_signal, 0, len(speed) - 1)
        ax1.scatter(valid_frames[idx_in_signal], speed[idx_in_signal],
                    color=color, zorder=5, label=src, s=50)
    ax1.set_ylabel("Wrist speed (per real frame)")
    ax1.set_title("Contact Detection V2 -- colored by confidence source")
    ax1.legend()

    ax2.plot(valid_frames, elbow_angles, label="Elbow angle", color="tab:green", alpha=0.6)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Elbow angle (degrees)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/contact_detection_v2.png")
    print("Chart saved to outputs/contact_detection_v2.png")


if __name__ == "__main__":
    main()
