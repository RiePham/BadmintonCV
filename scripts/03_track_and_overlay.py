"""
================================================================================
FILE 03: track_and_overlay.py
================================================================================

HOW TO RUN THIS FILE:
    python scripts/03_track_and_overlay.py \
        runs/player_only/weights/best.pt \
        runs/shuttlecock_only/weights/best.pt \
        videos/your_clip.mp4
        
    open outputs/tracked_two_models.mp4

    (3 things you must give it: the player robot, the shuttlecock robot,
    and the video you want to analyze)

WHAT THIS FILE DOES (super simple explanation):
    This file takes a badminton video, and for EVERY single frame, asks:
        1. The player robot: "where are the players in this frame?"
        2. The shuttlecock robot: "where is the shuttlecock in this frame?"
    Then it draws a green box around each player and a yellow dot on the
    shuttlecock, and saves a brand new video with those drawings on it.
    It ALSO writes down every detection (frame number, what was found,
    where) into a spreadsheet (CSV file) for later steps to use.

WHAT GOES IN, WHAT COMES OUT:
    IN:  runs/player_only/weights/best.pt       <- from 02_train_player.py
    IN:  runs/shuttlecock_only/weights/best.pt  <- from 02_train_shuttlecock.py
    IN:  videos/your_clip.mp4                   <- any badminton video
    OUT: outputs/tracked_two_models.mp4         <- video WITH boxes/dot drawn on it
    OUT: data/detections.csv                    <- spreadsheet of every detection

HOW THIS FILE CONNECTS TO OTHER FILES:
    02_train_player.py + 02_train_shuttlecock.py
              |
              v
    03_track_and_overlay.py   <-- YOU ARE HERE
              |
              v
    data/detections.csv
              

    NOTE: this file is a SEPARATE branch from the pose/PyTorch branch
    (files 05-10). It uses the shuttlecock's flight path, not a player's
    body pose, to guess when hits happen.
================================================================================
"""

import sys
import csv

import cv2
from ultralytics import YOLO

PLAYER_BOX_COLOR = (80, 220, 80)     # green box color for players
SHUTTLE_DOT_COLOR = (0, 215, 255)    # yellow dot color for the shuttlecock
SHUTTLE_DOT_RADIUS = 8
ROLE_NAMES = ["Player Near", "Player Far"]  # labels for the 2 players on screen


def main(player_model_path, shuttle_model_path, video_path, output_video_path, output_csv_path):
    # Load both trained robots. Each one only knows how to find ONE thing.
    player_model = YOLO(player_model_path)
    shuttle_model = YOLO(shuttle_model_path)

    # Open the video file so we can read it frame by frame, like flipping
    # through a flipbook one page at a time.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)           # how many frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up a writer that will build our NEW output video, frame by frame
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Set up a spreadsheet (CSV) writer to record every detection we find
    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "class_name", "confidence", "x_center", "y_center", "width", "height"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames left, video is done

        # WHY TWO SEPARATE .track() CALLS INSTEAD OF ONE:
        # Since these are two SEPARATE robots (not one robot that knows both
        # things), we have to ask each robot separately, on the same photo,
        # and then combine what they each found when we draw the picture.
        # This is a bit slower (we ask twice instead of once) but it lets
        # each robot be tuned just for its own job.
        #
        # persist=True tells each robot "remember what you saw in the last
        # frame" so it can keep track of the SAME player/shuttlecock across
        # frames instead of treating every frame like a brand new mystery.
        player_result = player_model.track(frame, persist=True, verbose=False)[0]
        shuttle_result = shuttle_model.track(frame, persist=True, conf=0.7, verbose=False)[0]

        # --- Handle player detections ---
        player_boxes = []
        for box in player_result.boxes:
            conf = float(box.conf[0])  # how sure the robot is (0 to 1)
            x_center, y_center, w, h = box.xywh[0].tolist()
            player_boxes.append((x_center, y_center, w, h))
            csv_writer.writerow([frame_idx, "player", round(conf, 3),
                                  round(x_center, 1), round(y_center, 1), round(w, 1), round(h, 1)])

        # A player standing further "down" on screen (bigger y) is closer to
        # the camera. Sort so the closest player is first -- this is how we
        # decide who to label "Near" vs "Far".
        player_boxes.sort(key=lambda p: p[1], reverse=True)
        for i, (x_center, y_center, w, h) in enumerate(player_boxes[:2]):
            label = ROLE_NAMES[i] if i < len(ROLE_NAMES) else f"Player {i+1}"
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_BOX_COLOR, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_BOX_COLOR, 2)

        # --- Handle shuttlecock detections ---
        # Only draw ONE dot per frame: whichever detection the robot is MOST
        # confident about. (No trail/history drawn -- just the current spot.)
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

        # Save this frame (with the boxes/dot drawn on it) into the new video
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    # Close everything cleanly when we're done
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