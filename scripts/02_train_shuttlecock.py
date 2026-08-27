"""
================================================================================
FILE 02b: train_shuttlecock.py
================================================================================

HOW TO RUN THIS FILE:
    python scripts/02_train_shuttlecock.py

WHAT THIS FILE DOES (super simple explanation):
    Same idea as 02_train_player.py, but this one teaches a SEPARATE robot
    to find the SHUTTLECOCK instead of players.

    Why a separate robot instead of one robot that finds both?
    The shuttlecock is TINY and moves SUPER fast -- much harder to spot than
    a whole player. Giving it its own dedicated robot, with a bigger model
    ("yolov8s" instead of "yolov8n") and bigger image size (1280 instead of
    640), lets it focus 100% on this one hard problem instead of splitting
    its attention with player-finding too.

WHAT GOES IN, WHAT COMES OUT:
    IN:  dataset_shuttlecock/data.yaml   <- made by 01_split_dataset_by_class.py
    OUT: runs/shuttlecock_only/weights/best.pt   <- the trained shuttlecock-finding robot

HOW THIS FILE CONNECTS TO OTHER FILES:
    01_split_dataset_by_class.py
              |
              v
    dataset_shuttlecock/data.yaml
              |
              v
    02_train_shuttlecock.py   <-- YOU ARE HERE
              |
              v
    runs/shuttlecock_only/weights/best.pt
              |
              v
    03_track_and_overlay.py   (uses this trained robot to find the shuttlecock in video)
================================================================================
"""

from ultralytics import YOLO

# "yolov8s" = "small" -- one size bigger than the "nano" used for players.
# The shuttlecock is a much harder, smaller target, so it gets a slightly
# stronger (but still lightweight) robot brain.
model = YOLO("yolov8s.pt")

results = model.train(
    data="dataset_shuttlecock/data.yaml",  # where the shuttlecock-only photos + labels are
    epochs=50,
    imgsz=1280,        # BIGGER image size than the player model -- helps spot a tiny fast object
    batch=8,           # lower this (e.g. 4) if you run out of memory
    project="runs",
    name="shuttlecock_only",
    patience=10,
)

print("\nTraining complete.")
print(f"Best weights saved at: runs/shuttlecock_only/weights/best.pt")
print(f"Validation metrics (mAP, precision, recall) are in: runs/shuttlecock_only/results.csv")