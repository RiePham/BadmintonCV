from ultralytics import YOLO
model = YOLO("runs/detect/runs/shuttlecock_only/weights/last.pt")
model.train(resume=True)