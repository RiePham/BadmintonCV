"""
Run TWO separate models on the same video: one trained only on player,
one trained only on shuttlecock. Each model runs independently per frame,
and results are drawn onto the same output video.

WHY TWO MODEL.TRACK() CALLS PER FRAME INSTEAD OF ONE:
Since player_model and shuttle_model are now separate networks (not one
2-class model), there's no single call that returns both. So each frame,
we hand the same raw frame to both models separately and combine their
outputs when drawing. This is slower per frame (~2x inference time) than
one shared model, but each individual model can be tuned independently
(different imgsz, different architecture size) for its own difficulty level.

HOW TO RUN:
    python scripts/03_track_and_overlay.py \
        runs/detect/runs/player_only/weights/best.pt \
        runs/detect/runs/shuttlecock_only/weights/best.pt \
        videos/your_clip.mp4
"""

import sys
import csv

import cv2
from ultralytics import YOLO

PLAYER_BOX_COLOR = (80, 220, 80)
SHUTTLE_DOT_COLOR = (0, 215, 255)
SHUTTLE_DOT_RADIUS = 8
ROLE_NAMES = ["Player Near", "Player Far"]


def main(player_model_path, shuttle_model_path, video_path, output_video_path, output_csv_path):
    player_model = YOLO(player_model_path)
    shuttle_model = YOLO(shuttle_model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "class_name", "confidence", "x_center", "y_center", "width", "height"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # persist=True keeps each model's own tracking state alive between
        # calls, even though we're calling them frame-by-frame here rather
        # than handing over the whole video at once.
        player_result = player_model.track(frame, persist=True, verbose=False)[0]
        shuttle_result = shuttle_result = shuttle_model.track(frame, persist=True, conf=0.7, verbose=False)[0]

        # --- Players: box + Near/Far role, same logic as before ---
        player_boxes = []
        for box in player_result.boxes:
            conf = float(box.conf[0])
            x_center, y_center, w, h = box.xywh[0].tolist()
            player_boxes.append((x_center, y_center, w, h))
            csv_writer.writerow([frame_idx, "player", round(conf, 3),
                                  round(x_center, 1), round(y_center, 1), round(w, 1), round(h, 1)])

        player_boxes.sort(key=lambda p: p[1], reverse=True)  # larger y = closer to camera
        for i, (x_center, y_center, w, h) in enumerate(player_boxes[:2]):
            label = ROLE_NAMES[i] if i < len(ROLE_NAMES) else f"Player {i+1}"
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_BOX_COLOR, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_BOX_COLOR, 2)

        # --- Shuttlecock: single dot at the highest-confidence detection, no trail ---
        best_shuttle = None
        for box in shuttle_result.boxes:
            conf = float(box.conf[0])
            x_center, y_center, w, h = box.xywh[0].tolist()
            csv_writer.writerow([frame_idx, "shuttlecock", round(conf, 3),
                                  round(x_center, 1), round(y_center, 1), round(w, 1), round(h, 1)])
            if best_shuttle is None or conf > best_shuttle[0]:
                best_shuttle = (conf, x_center, y_center)

        if best_shuttle is not None:
            _, x, y = best_shuttle
            cv2.circle(frame, (int(x), int(y)), SHUTTLE_DOT_RADIUS, SHUTTLE_DOT_COLOR, -1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    csv_file.close()
    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Video: {output_video_path}")
    print(f"Detections CSV: {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/03_track_and_overlay.py <player_model.pt> <shuttle_model.pt> <video.mp4>")
        sys.exit(1)

    main(
        player_model_path=sys.argv[1],
        shuttle_model_path=sys.argv[2],
        video_path=sys.argv[3],
        output_video_path="outputs/tracked_two_models.mp4",
        output_csv_path="data/detections.csv",
    )