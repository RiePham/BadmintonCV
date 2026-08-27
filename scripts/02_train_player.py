"""
================================================================================
FILE 02a: train_player.py
================================================================================

HOW TO RUN THIS FILE:
    python scripts/02_train_player.py

WHAT THIS FILE DOES (super simple explanation):
    This file teaches a robot (a YOLOv8 model) to find PLAYERS in photos.
    The robot isn't starting from zero -- it already watched millions of
    photos before (of people, cars, dogs, etc.) so it already kind of knows
    what a "person shape" looks like. We just show it OUR specific photos
    (badminton players) so it gets really good at finding THEM specifically.
    This "starting smart, then practicing on our photos" trick is called
    FINE-TUNING, and it's much faster than teaching a robot from nothing.

WHAT GOES IN, WHAT COMES OUT:
    IN:  dataset_player/data.yaml   <- made by 01_split_dataset_by_class.py
    OUT: runs/player_only/weights/best.pt   <- the trained player-finding robot

HOW THIS FILE CONNECTS TO OTHER FILES:
    01_split_dataset_by_class.py
              |
              v
    dataset_player/data.yaml
              |
              v
    02_train_player.py   <-- YOU ARE HERE
              |
              v
    runs/player_only/weights/best.pt
              |
              v
    03_track_and_overlay.py   (uses this trained robot to find players in video)
================================================================================
"""

from ultralytics import YOLO

# "yolov8n" = the smallest, fastest version of YOLO ("n" = nano).
# We pick the small one because we don't have a huge dataset or a huge
# computer -- small and fast is the right choice here.
# The first time this line runs, it downloads the pretrained brain
# (already trained on 80 everyday objects) automatically from the internet.
model = YOLO("yolov8n.pt")

# This is the actual "teaching" step. Think of it like homework:
# the robot looks at photos over and over (epochs), checks its answers,
# and slowly gets better at finding players.
results = model.train(
    data="dataset_player/data.yaml",  # where the player-only photos + labels are
    epochs=50,        # how many times the robot studies the whole photo set
    imgsz=640,         # photos get resized to 640x640 pixels before studying
    batch=8,           # how many photos it studies at once (lower this if your computer runs out of memory)
    project="runs",    # top folder where results get saved
    name="player_only",  # sub-folder name for THIS training run
    patience=10,       # if the robot stops improving for 10 epochs in a row, stop early (saves time)
)

print("\nTraining complete.")
print(f"Best weights saved at: runs/player_only/weights/best.pt")
print(f"Validation metrics (mAP, precision, recall) are in: runs/player_only/results.csv")