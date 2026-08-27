"""
================================================================================
FILE 08: calculate_contacts_v2.py  ("v2" -- the FIXED, better version of file 07)
================================================================================

HOW TO RUN THIS FILE:
    python scripts/08_calculate_contacts_v2.py

WHAT THIS FILE DOES (super simple explanation):
    Same GOAL as file 07: guess when the player hit the shuttlecock, by
    watching how fast their wrist moves. But file 07 got fooled a lot --
    it thought a hit happened when really it was just noise (the camera
    briefly losing track of the wrist) or a math mistake (treating several
    skipped frames as if they were just ONE super-fast frame).

    This file fixes those mistakes with 4 tricks:
      A) Ignore frames where MediaPipe wasn't confident about the wrist spot
      B) Smooth the wrist's PATH first, before measuring speed (not after)
      C) Only count it as a swing if the arm was STILL first, then suddenly
         burst into fast motion (a real swing looks like this; slow
         repositioning does not)
      D) Fix the "missing frames" math bug -- measure speed per REAL frame
         gap, not just "this reading vs the very last reading saved"
    It can ALSO (optionally) listen to the video's audio for the "pok"
    sound of a real hit, and use that to double-check its guesses.

WHAT GOES IN, WHAT COMES OUT:
    IN:  data/pose_landmarks.csv      <- made by 06_pose_extraction.py
    OUT: outputs/contact_detection_v2.png  <- a chart
    OUT: data/shot_features_v2.csv    <- list of guessed hit moments (v2, GOOD one)

HOW THIS FILE CONNECTS TO OTHER FILES:
    06_pose_extraction.py
              |
              v
    data/pose_landmarks.csv
              |
              v
    08_calculate_contacts_v2.py   <-- YOU ARE HERE (replaces file 07)
              |
              v
    data/shot_features_v2.csv
              |
        (you manually watch the video and fill in the "shot_type" column
         by hand -- e.g. "smash", "push", "not_shot")
              |
              v
    09_train_shot_classifier.py   (teaches a small robot to guess shot_type
                                   from the numbers in this spreadsheet)
================================================================================

**Why V1 was wrong:**
- The camera sometimes "guessed wrong" where the hand was when it moved super fast (like blurry eyes)
- If the camera missed a few frames and lumped them together, it looked like the hand moved super fast — but it didn't really
- V1 never checked "was the hand still first?" — it just trusted anything fast

**How V2 fixes it:**
1. If the camera isn't sure, don't trust that frame
2. Count the missed frames correctly, so it can't be fooled anymore
3. Only count it as a real hit if the hand was still first, then burst fast
4. (Extra, optional) Also listen for the "pok" sound of contact, to double-check

**V2 still isn't perfect:** it still can't watch the shuttlecock's flight path — that's the most accurate way, but it needs a separate robot to track the shuttlecock, which isn't built yet.
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
FPS = 25  # your badminton clip is 25fps -- keep this in sync with the source video
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
    """
    SIMPLE EXPLANATION:
    Find one specific joint's (x, y) spot in one frame's data. If MediaPipe
    wasn't confident about it (visibility too low), pretend we don't have
    it at all -- better to have NO data than WRONG data.
    """
    row = frame_df[frame_df["landmark_name"] == name]
    if row.empty:
        return None
    if "visibility" in row.columns:
        vis = row["visibility"].values[0]
        if pd.notna(vis) and vis < min_visibility:
            return None  # MediaPipe itself wasn't confident -- don't trust this frame
    return np.array([row["x"].values[0], row["y"].values[0]])


def calculate_angle(a, b, c):
    """
    SIMPLE EXPLANATION:
    Measures how bent or straight the elbow is, using the 3 points
    shoulder(a), elbow(b), wrist(c). Straight arm = close to 180 degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def smooth_track(positions):
    """
    SIMPLE EXPLANATION:
    Takes a wobbly path of (x, y) points and makes it a smoother, cleaner
    curve -- like tracing over a shaky hand-drawn line with a steadier
    hand. This uses a common smoothing method (Savitzky-Golay).
    """
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
    """
    SIMPLE EXPLANATION:
    Same smoothing as above, but done in careful little chunks. If some
    frames were dropped in the middle (a "gap"), we DON'T smooth straight
    across that gap -- that would blend real motion with the missing
    frames and give a wrong answer. Instead we smooth each unbroken chunk
    separately.

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
    """
    SIMPLE EXPLANATION:
    This is the main "reading" function. It goes through the whole
    spreadsheet from file 06, grabs the shoulder/elbow/wrist for every
    frame, smooths the paths, and calculates two things we care about for
    every frame: (1) the elbow angle, and (2) how fast the wrist is moving
    (speed), fixed so it correctly accounts for any dropped frames.
    """
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
    """
    SIMPLE EXPLANATION:
    Looks through the whole speed signal and finds "bumps" (peaks). But
    unlike file 07, it doesn't stop there -- for EACH bump, it checks
    backward in time: "was the wrist basically still right before this
    bump, and did it speed up FAST and SHARPLY?" Only bumps that pass this
    check are kept as real swing candidates. Slow, gradual speed-ups
    (like just walking/repositioning) get thrown out even if they reach a
    locally-high speed at some point.
    """
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

    # Compare all candidates against each other: keep a note of which ones
    # had a REALLY sharp burst (top 60%) vs a weaker one (bottom 40%) --
    # this becomes the "pose_sharp" vs "pose_weak" confidence label later.
    rates = np.array([c["burst_rate"] for c in candidates])
    rate_floor = np.percentile(rates, BURST_RATE_PERCENTILE)
    for c in candidates:
        c["sharp_burst"] = c["burst_rate"] >= rate_floor

    return candidates


# ------------------------------------------------------------------------
# Optional audio fusion
# ------------------------------------------------------------------------
def extract_audio(video_path, out_wav="data/_extracted_audio.wav"):
    """SIMPLE EXPLANATION: pulls just the sound track out of a video file
    and saves it as a plain audio (.wav) file, using the ffmpeg tool."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "22050", out_wav],
        check=True, capture_output=True,
    )
    return out_wav


def get_audio_onset_frames(video_path, fps):
    """
    SIMPLE EXPLANATION:
    Listens to the video's audio track and finds moments where a SHARP,
    SUDDEN sound happens (like the "pok" of a racket hitting the
    shuttlecock). This is OPTIONAL extra evidence -- if a candidate swing
    from the pose data lines up with a sharp sound, we trust it more.
    """
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
    """
    SIMPLE EXPLANATION:
    For each pose-based candidate swing, check if there's a matching sharp
    sound nearby in time. If yes, use the (more precise) audio timestamp
    instead of the pose one.
    """
    if audio_onset_frames is None or len(audio_onset_frames) == 0:
        return [None] * len(candidate_frame_numbers)

    matches = []
    for cf in candidate_frame_numbers:
        diffs = np.abs(audio_onset_frames - cf)
        best = np.argmin(diffs)
        matches.append(int(audio_onset_frames[best]) if diffs[best] <= tolerance else None)
    return matches


OUTPUT_CSV = "data/shot_features_v2.csv"


def load_existing_labels(path):
    """
    SIMPLE EXPLANATION:
    Every time you run this file again, it recalculates everything from
    scratch -- which would normally ERASE any "shot_type" labels
    (smash/push/not_shot) you already typed in by hand. This function
    reads the OLD file first and remembers those labels (matched by exact
    frame number), so we can put them back afterward instead of losing
    your work.
    """
    if not os.path.exists(path):
        return {}
    try:
        old = pd.read_csv(path)
    except Exception:
        return {}
    if "shot_type" not in old.columns:
        return {}
    labels = {}
    for _, row in old.iterrows():
        label = row["shot_type"]
        if pd.notna(label) and str(label).strip():
            labels[int(row["frame"])] = str(label).strip()
    return labels


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    """
    SIMPLE EXPLANATION -- the overall recipe this file follows:
        1. Remember any labels you already typed in by hand
        2. Read the joint positions (file 06's CSV) and calculate speed/angle
        3. Find candidate swings (still -> burst pattern)
        4. (Optional) Check candidates against audio "pok" sounds
        5. Save everything to a new spreadsheet, keeping your old labels
        6. Draw a chart so you can visually check the results
    """
    existing_labels = load_existing_labels(OUTPUT_CSV)
    if existing_labels:
        print(f"Preserving {len(existing_labels)} existing shot_type label(s) "
              f"from a previous run (matched by exact frame number)")

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
        # Every candidate gets a "source" label showing how much we trust it:
        #   pose+audio  = pose AND sound agree (most trustworthy)
        #   pose_sharp  = pose alone, but a strong clear burst
        #   pose_weak   = pose alone, a weaker/less clear burst (double check by eye)
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
            "shot_type": existing_labels.get(final_frame, ""),
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
                "shot_type": existing_labels.get(int(af), ""),
            })

    features_df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)

    unmatched_labels = set(existing_labels) - set(features_df["frame"])
    if unmatched_labels:
        print(f"WARNING: {len(unmatched_labels)} existing label(s) had no matching "
              f"frame in this run and were dropped (frames: {sorted(unmatched_labels)}) "
              f"-- the detector likely found a different set of candidates this time.")

    features_df.to_csv(OUTPUT_CSV, index=False)

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