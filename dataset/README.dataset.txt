# badminton-players-detection > 2026-08-26 4:55pm
https://universe.roboflow.com/le-minh-dat-pham/badminton-players-detection-gwgb1-vuwhi

Provided by a Roboflow user
License: CC BY 4.0

## 📌 Project Overview

This project aims to build an object detection model capable of identifying **badminton players** and **shuttlecocks** in images and video frames.
* All images in the dataset are taken from badminton matches captured in two perspectives:
	* Broadcast view (similar to TV coverage, usually from an elevated angle).
	* Ground view – recorded by individuals using their own devices (e.g., mobile phones, handheld cameras) while standing at court level.
* The dataset images are extracted from public YouTube videos of matches.
* All images are standardized at a resolution of 1920×1080 (Full HD) for consistency.

By training a robust dataset and refining annotations, the model can support applications such as:
* Automated match analysis and statistics tracking
* Player movement heatmaps, speed, distance and strategy insights
* Shuttlecock trajectory tracking for highlight generation

## 🏷️ Class Descriptions

* **player**: Represents a badminton athlete visible in the frame. Players may appear in various poses (smashing, serving, defending, moving, etc.), so bounding boxes should cover the full body.
* **shuttlecock**: Represents the badminton shuttlecock (also known as birdie). This object is often small, fast-moving, and partially blurred in frames. Bounding boxes should be as tight as possible around the shuttlecock.

## 🤝 Labeling Guidelines

* Draw bounding boxes tightly around each object.
* Ensure all players in the frame are labeled.
* The shuttlecock should always be annotated, even if partially blurred or in motion.
* Avoid overlapping boxes unless absolutely necessary.