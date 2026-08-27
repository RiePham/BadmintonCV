"""
================================================================================
FILE 05: test_pose.py  (a quick "does this even work?" test -- see note below)
================================================================================

HOW TO RUN THIS FILE:
    python scripts/05_test_pose.py

WHAT THIS FILE DOES (super simple explanation):
    This is a tiny practice run, NOT a real part of the pipeline.
    It grabs just the FIRST 200 frames of one video, draws green dots on
    the body joints it finds (nose, shoulders, elbows, wrists, etc.), and
    saves only ONE picture (the very last frame it looked at) so you can
    quickly check: "does MediaPipe actually detect the person's body in
    this video?" before spending time building the real pipeline.

WHAT GOES IN, WHAT COMES OUT:
    IN:  videos/videomomota.mp4  (hard-coded filename -- change it if needed)
    OUT: test_pose_frame.jpg     <- just ONE image, for a quick look

HOW THIS FILE CONNECTS TO OTHER FILES:
    This file does NOT connect to anything else. Nothing later in the
    pipeline reads test_pose_frame.jpg. It was just a sanity check before
    building 06_pose_extraction.py.

    05_test_pose.py  -->  (dead end, just a manual visual check)

DIFFERENCE FROM FILE 06 (06_pose_extraction.py):
    - This file (05): only 200 frames, saves only 1 image, no CSV data saved.
      Good for a 10-second gut check.
    - File 06: runs on the ENTIRE video, saves a full video with skeleton
      drawn on every frame, AND saves a CSV with every joint's (x, y, z,
      visibility) for EVERY frame -- this CSV is the real data used later
      by files 07/08/09/10.

SHOULD YOU DELETE THIS FILE?
    Yes, it's safe to delete -- it was only a one-time test and nothing
    else in the project depends on it. If you want to keep it just in
    case, that's also fine; it won't break anything either way.
================================================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the pose-detecting "brain" (a pretrained model file) and tell it to
# run on the CPU (not the graphics card/GPU) -- this avoids a known crash
# on Apple Silicon Macs when using the GPU/Metal path.
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task",
    delegate=python.BaseOptions.Delegate.CPU,  # tránh lỗi GPU/Metal trên macOS
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # tell it we're feeding a video, frame by frame
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Open the video and figure out how fast it plays (frames per second)
cap = cv2.VideoCapture("videos/videomomota.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
frame = None
# Only look at the first 200 frames -- just enough to sanity-check things
while frame_idx < 200:
    ret, frame = cap.read()
    if not ret:
        break  # video is shorter than 200 frames, that's fine, just stop

    # MediaPipe wants pictures in RGB color order, but OpenCV reads them in
    # BGR order -- this line converts it so the colors aren't swapped.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # MediaPipe (in VIDEO mode) needs to know the timestamp of each frame,
    # in milliseconds, so it can track motion smoothly between frames.
    timestamp_ms = int(frame_idx * (1000 / fps))

    # Ask the pose model: "what body points do you see in this frame?"
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        h, w, _ = frame.shape
        for landmarks in result.pose_landmarks:
            for lm in landmarks:
                # lm.x and lm.y are given as fractions (0.0 to 1.0) of the
                # image size, so we multiply by width/height to get real
                # pixel positions to draw a dot at.
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    frame_idx += 1

cap.release()
# Only the LAST frame looked at gets saved as a picture -- just enough to
# eyeball whether the green dots landed on the right body parts.
cv2.imwrite("test_pose_frame.jpg", frame)
print("Xong, mở test_pose_frame.jpg để xem kết quả")