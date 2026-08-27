"""
================================================================================
FILE 06: pose_extraction.py
================================================================================

HOW TO RUN THIS FILE:
    python scripts/06_pose_extraction.py videos/testvid2.mp4
    (swap in whatever video file you want to analyze)

WHAT THIS FILE DOES (super simple explanation):
    This file watches a WHOLE badminton video, frame by frame, and for
    every frame it asks MediaPipe: "where are this person's body joints
    right now?" (nose, shoulders, elbows, wrists, hips, knees, etc. -- 33
    points total). It draws a green dot on each joint and saves a new
    video with all those dots. MORE IMPORTANTLY, it writes down the exact
    (x, y, z, visibility) numbers for every joint on every frame into a
    big spreadsheet (CSV file) -- this spreadsheet is the REAL DATA that
    every later step in the "pose" branch of the project depends on.

WHAT GOES IN, WHAT COMES OUT:
    IN:  videos/testvid2.mp4 (or whichever video you pass in)
    OUT: outputs/pose_output.mp4     <- video with green dots drawn on
    OUT: data/pose_landmarks.csv     <- spreadsheet: joint positions every frame

HOW THIS FILE CONNECTS TO OTHER FILES:
    06_pose_extraction.py   <-- YOU ARE HERE
              |
              v
    data/pose_landmarks.csv
              |
              v
    07_calculate_angles.py  (v1 -- older, has known problems)
    08_calculate_contacts_v2.py  (v2 -- the better, fixed version)
              |
              v
    data/shot_features_v2.csv
              |
              v
    09_train_shot_classifier.py  (teaches a small PyTorch model to guess shot type)
              |
              v
    10_run_on_video.py  (runs the WHOLE chain automatically on a new video)
================================================================================
"""

import sys
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe always returns exactly 33 body points, always in this same
# order. This list just gives each point a human-readable name (instead
# of just a number 0-32) so the CSV is easy to read later.
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


def main(video_path, output_video_path, output_csv_path):
    """
    SIMPLE EXPLANATION:
    This function does the whole job: open the video, set up the pose
    detector, go through every frame, save the results.
    """
    # Load the pose-detecting "brain" and run it on CPU (avoids a known
    # crash with the GPU/Metal path on Apple Silicon Macs).
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker.task",
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # we're feeding a video, not single photos
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Open the input video and read its basic info (speed, size)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the writer for the NEW output video (with dots drawn on it)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Set up the spreadsheet (CSV) we'll fill in with every joint's position
    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "landmark_id", "landmark_name", "x", "y", "z", "visibility"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames, video is finished

        # MediaPipe needs RGB colors, OpenCV gives us BGR -- convert it
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Tell MediaPipe exactly what time (in milliseconds) this frame
        # happens at, so it can track motion smoothly across frames.
        timestamp_ms = int(frame_idx * (1000 / fps))

        # Ask MediaPipe: "what body points do you see in this frame?"
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                for i, lm in enumerate(landmarks):
                    # lm.x, lm.y are fractions of the image size (0.0-1.0).
                    # Multiply by width/height to get a real pixel spot to
                    # draw a green dot at.
                    x_px, y_px = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)
                    # Also save the RAW numbers (not pixels) to the CSV --
                    # this is what later scripts (07/08/09/10) actually use.
                    csv_writer.writerow([
                        frame_idx, i, LANDMARK_NAMES[i],
                        round(lm.x, 4), round(lm.y, 4), round(lm.z, 4),
                        round(lm.visibility, 4)
                    ])

        # Save this frame (with dots drawn on it) into the new output video
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Đã xử lý {frame_idx} frames...")

    # Close everything cleanly
    cap.release()
    writer.release()
    csv_file.close()
    print(f"\nXong. {frame_idx} frames.")
    print(f"Video: {output_video_path}")
    print(f"Landmarks CSV: {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/06_pose_extraction.py <đường_dẫn_video>")
        sys.exit(1)

    main(
        video_path=sys.argv[1],
        output_video_path="outputs/pose_output.mp4",
        output_csv_path="data/pose_landmarks.csv",
    )