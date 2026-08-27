import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task",
    delegate=python.BaseOptions.Delegate.CPU,  # tránh lỗi GPU/Metal trên macOS
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("videos/videomomota.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
frame = None
while frame_idx < 200:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(frame_idx * (1000 / fps))

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        h, w, _ = frame.shape
        for landmarks in result.pose_landmarks:
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    frame_idx += 1

cap.release()
cv2.imwrite("test_pose_frame.jpg", frame)
print("Xong, mở test_pose_frame.jpg để xem kết quả")