# Vehicle Detection & Tracking System using YOLOv8

A real-time object detection and tracking system built with **YOLOv8** and **OpenCV** that detects vehicles in traffic footage, tracks them frame-to-frame with persistent IDs, and counts vehicles as they cross a defined line — all while overlaying live processing statistics on the output video.

---

## Overview

Traffic monitoring is a common real-world computer vision application used in smart city systems, toll booths, and traffic analytics platforms. This project simulates a simplified version of such a system: given a video feed of highway traffic, it detects objects, assigns each a persistent tracking ID, and counts how many objects cross a virtual line — a core building block for vehicle counting and flow analysis systems.

## Features

- **Real-time object detection** using the YOLOv8 (`yolov8n`) model
- **Multi-object tracking** with persistent IDs across frames (via Ultralytics' built-in tracker)
- **Line-crossing vehicle counter** — counts objects as they cross a configurable horizontal line
- **Live FPS overlay** to monitor processing performance
- **Robust error handling** for missing video files, missing model weights, and failed video writers
- **Clean, modular code structure** — organized into reusable functions rather than a single script block
- **Configurable parameters** — confidence threshold, line position, paths, all centralized at the top of the script

## Demo Output

The script processes an input video and produces an annotated output video showing:
- Bounding boxes with class labels and tracking IDs
- A counting line
- Live vehicle count and FPS in the top-left corner

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Tracking | Ultralytics built-in tracker (ByteTrack) |
| Video Processing | OpenCV |
| Language | Python 3 |

## Project Structure

```
.
├── vehicle_detection_tracking.py   # Main script
├── yolov8n.pt                      # YOLOv8 nano model weights
├── highway_traffic.mp4             # Input video (sample traffic footage)
└── detection_output.mp4            # Generated output (after running the script)
```

## Installation

```bash
pip install ultralytics opencv-python
```

## Usage

1. Place your input video in the project directory and update `INPUT_VIDEO_PATH` in the script if needed.
2. Run the script:

```bash
python vehicle_detection_tracking.py
```

3. Press `q` at any time to stop processing early.
4. The annotated output video will be saved as `detection_output.mp4`, and the total vehicle count will be printed to the console.

## Configuration

All key parameters are defined at the top of the script:

```python
INPUT_VIDEO_PATH = "highway_traffic.mp4"
OUTPUT_VIDEO_PATH = "detection_output.mp4"
MODEL_PATH = "yolov8n.pt"
COUNT_LINE_POSITION = 0.6   # fraction of frame height
CONFIDENCE_THRESHOLD = 0.4
```

## How It Works

1. **Detection & Tracking** — Each frame is passed through YOLOv8 with `persist=True`, which maintains consistent object IDs across frames instead of treating every frame as an independent detection.
2. **Counting Logic** — For every tracked object, the script stores its previous vertical center position. If an object's center moves from above the counting line to below it (or vice versa) between frames, the counter increments.
3. **Overlay & Output** — Each processed frame is annotated with bounding boxes, the counting line, live vehicle count, and current FPS, then written to the output video file.

## Future Improvements

- Class-specific filtering (e.g., count only cars, trucks, and buses)
- Direction-aware counting (separate counts for inbound vs. outbound traffic)
- Speed estimation using frame-to-frame displacement and a calibrated pixel-to-distance ratio
- Multi-lane counting zones for highway segments
- Export counting data to CSV/JSON for further analysis or dashboarding

## Author

Built as part of an ongoing portfolio of Data Science, Machine Learning, and Computer Vision projects.
