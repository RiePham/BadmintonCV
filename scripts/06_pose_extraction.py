"""
Chạy MediaPipe Pose trên toàn bộ video, vẽ skeleton lên video output,
và ghi lại toạ độ mọi khớp mỗi frame vào CSV cho bước tính góc ở Ngày 2.

Chạy: python scripts/06_pose_extraction.py videos/testvid2.mp4
"""

import sys
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker.task",
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "landmark_id", "landmark_name", "x", "y", "z", "visibility"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / fps))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                for i, lm in enumerate(landmarks):
                    x_px, y_px = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)
                    csv_writer.writerow([
                        frame_idx, i, LANDMARK_NAMES[i],
                        round(lm.x, 4), round(lm.y, 4), round(lm.z, 4),
                        round(lm.visibility, 4)
                    ])

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Đã xử lý {frame_idx} frames...")

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