"""
Vehicle Detection & Tracking System
------------------------------------
Detects and tracks objects in a traffic video using YOLOv8, counts vehicles
crossing a defined line, and overlays live statistics (count, FPS) on the
output video.

"""

import os
import time

import cv2
from ultralytics import YOLO

# ----------------------------- Configuration ------------------------------
INPUT_VIDEO_PATH = "highway traffic.mp4"
OUTPUT_VIDEO_PATH = "detection_output.mp4"
MODEL_PATH = "yolov8n.pt"

# Position of the counting line as a fraction of frame height (0.0 - 1.0).
# 0.5 means the line is drawn halfway down the frame.
COUNT_LINE_POSITION = 0.6

CONFIDENCE_THRESHOLD = 0.4
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> YOLO:
    """Load the YOLO model, raising a clear error if it fails."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Make sure the .pt file is in the correct directory."
        )
    return YOLO(model_path)


def open_video(input_path: str) -> cv2.VideoCapture:
    """Open the input video and validate that it was read successfully."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found at '{input_path}'.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file '{input_path}'.")
    return cap


def create_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """Create a VideoWriter matching the input video's properties."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if FPS metadata is missing

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise IOError(f"Could not create output video writer for '{output_path}'.")

    return writer


def has_crossed_line(prev_y: float, curr_y: float, line_y: int) -> bool:
    """Check whether a tracked object's centre crossed the counting line."""
    return prev_y < line_y <= curr_y


def draw_overlay(frame, count: int, fps: float, line_y: int, width: int):
    """Draw the counting line and live statistics on the frame."""
    cv2.line(frame, (0, line_y), (width, line_y), LINE_COLOR, LINE_THICKNESS)

    cv2.putText(
        frame, f"Vehicles Counted: {count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2
    )


def run_detection():
    model = load_model(MODEL_PATH)
    cap = open_video(INPUT_VIDEO_PATH)
    out = create_video_writer(cap, OUTPUT_VIDEO_PATH)

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_y = int(frame_height * COUNT_LINE_POSITION)

    vehicle_count = 0
    track_history = {}  # track_id -> last known centre y-coordinate

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # persist=True keeps consistent IDs across frames for tracking
            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            annotated = results[0].plot()

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xywh = boxes.xywh.cpu().tolist()

                for track_id, (x, y, w, h) in zip(ids, xywh):
                    centre_y = y + h / 2

                    if track_id in track_history:
                        prev_y = track_history[track_id]
                        if has_crossed_line(prev_y, centre_y, line_y):
                            vehicle_count += 1

                    track_history[track_id] = centre_y

            fps = 1.0 / (time.time() - start_time + 1e-6)
            draw_overlay(annotated, vehicle_count, fps, line_y, frame_width)

            out.write(annotated)
            cv2.imshow("Vehicle Detection & Tracking", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Total vehicles counted: {vehicle_count}")
        print(f"Output saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    run_detection()