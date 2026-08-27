"""
Fine-tune a pretrained YOLOv8 model on the badminton player+shuttlecock dataset.

WHY THIS APPROACH (be ready to explain this in the interview):
YOLOv8n comes pretrained on COCO (person, car, dog, etc. -- 80 general classes).
It already knows how to detect "person"-like shapes well. Instead of training a
detector from zero (which needs huge data + compute), we FINE-TUNE it: start from
the COCO-pretrained weights and continue training on our small, domain-specific
badminton dataset. The early layers (edges, textures, shapes) transfer almost for
free; only the later layers really need to adapt to "shuttlecock" as a new concept.
This is the standard, correct way to build a custom detector with a small dataset.

HOW TO RUN:
    1. Download the Roboflow dataset (YOLOv8 format), unzip it into ./dataset/
       so you have: dataset/data.yaml, dataset/train/, dataset/valid/, dataset/test/
    2. pip install ultralytics
    3. python scripts/02_train_detector.py
"""

from ultralytics import YOLO

# yolov8n = "nano", the smallest/fastest variant -- right choice for CPU/limited time.
# If training feels too slow, this is already the fastest option; reduce epochs instead.
model = YOLO("yolov8n.pt")  # downloads pretrained COCO weights automatically, once

results = model.train(
    data="dataset_player/data.yaml",   # path to the Roboflow-generated config
    epochs=50,                   # start here; watch the loss curve, stop early if it plateaus
    imgsz=640,
    batch=8,                     # lower this (e.g. 4) if you run out of memory
    project="runs",
    name="player_only",
    patience=10,                 # stop early if validation stops improving for 10 epochs
)

print("\nTraining complete.")
print(f"Best weights saved at: runs/badminton_finetune/weights/best.pt")
print(f"Validation metrics (mAP, precision, recall) are in: runs/badminton_finetune/results.csv")
