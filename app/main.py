"""
Week 2: Input + Detection + Tracking + Behavior Pipeline

Full Phase 1 pipeline with:
    1. Input: Capture from webcam or video file
    2. Detection: YOLOv8 person detection
    3. Tracking: DeepSORT multi-person tracking
    4. Behavior: Rule-based action classification

Pipeline:
    Frame → Detection → Tracking → Behavior → Display

Usage:
    # Webcam
    python app/main.py

    # Video file
    python app/main.py --source data/samples/test_video.mp4

Controls:
    q - quit
    p - pause/resume
    r - reset tracker
"""

import sys
import cv2
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_module.input_handler import InputModule
from src.detection_module.detector import DetectionModule
from src.tracking_module.tracker import TrackingModule
from src.behavior_module.behavior_analyzer import BehaviorAnalyzer
from src.utils.config_loader import load_config
from src.utils.visualization import (
    draw_tracks, draw_fps, draw_behavior_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Week 2: Detection + Tracking + Behavior Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: 0 for webcam, or path to video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display (for headless mode)"
    )
    return parser.parse_args()


def main():
    """Main pipeline: Input → Detection → Tracking → Behavior → Display."""
    args = parse_args()

    # Load configuration
    print("=" * 60)
    print("Week 2: Detection + Tracking + Behavior Pipeline")
    print("=" * 60)

    config = load_config(args.config)

    # Determine video source (CLI arg overrides config)
    source = args.source if args.source is not None else config.input.source
    if source == "0" or source == 0:
        source = 0
        print("Source: Webcam (0)")
    else:
        print(f"Source: {source}")

    # Initialize modules
    print("\n[1/4] Initializing InputModule...")
    input_module = InputModule(
        source=source,
        width=config.input.get("width", 640),
        height=config.input.get("height", 480),
        fps=config.input.get("fps", 30)
    )

    print("\n[2/4] Initializing DetectionModule...")
    detector = DetectionModule(
        model_path=config.detection.model_path,
        confidence_threshold=config.detection.confidence_threshold,
        classes=config.detection.classes,
        device=config.detection.device
    )
    detector.warmup()

    print("\n[3/4] Initializing TrackingModule...")
    tracker = TrackingModule(
        max_age=30,          # Keep track for 30 frames without detection
        min_hits=3,          # Confirm after 3 detections
        iou_threshold=0.3,   # Match if IoU > 0.3
        history_size=30      # Keep 30 past positions
    )

    print("\n[4/4] Initializing BehaviorModule...")
    behavior_analyzer = BehaviorAnalyzer(
        walking_velocity_range=(2.0, 8.0),
        running_velocity_threshold=8.0,
        falling_aspect_ratio_threshold=1.0
    )

    # FPS calculation
    fps_history = []
    fps_window = 30

    # Add frame skipping and resize logic
    frame_skip_counter = 0  # Initialize frame skip counter
    frame_skip_interval = 3  # Process every 3rd frame for better FPS

    print("\n" + "=" * 60)
    print("Pipeline ready!")
    print("Controls: q=quit | p=pause | r=reset tracker")
    print("=" * 60 + "\n")

    paused = False

    try:
        for frame in input_module.stream():
            # Skip frames to improve FPS
            frame_skip_counter += 1
            if frame_skip_counter % frame_skip_interval != 0:
                continue

            # Resize frame to reduce processing time
            frame.image = cv2.resize(frame.image, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Calculate FPS
            current_time = time.time()
            fps_history.append(current_time)
            if len(fps_history) > fps_window:
                fps_history.pop(0)

            fps = len(fps_history) / (fps_history[-1] - fps_history[0]) \
                if len(fps_history) > 1 else 0

            # Handle pause
            if paused:
                cv2.imshow("Week 2: Tracking + Behavior", frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = not paused
                continue

            # === PIPELINE ===

            # Step 1: Detection
            detections = detector.detect(frame.image)

            # Increase confidence threshold to reduce unnecessary computations
            detections = [d for d in detections if d.confidence > 0.6]

            # Step 2: Tracking (pass frame for appearance features)
            tracks = tracker.update(detections, frame=frame.image)

            # Step 3: Behavior Analysis
            behaviors = behavior_analyzer.analyze(tracks)
            behavior_map = {b.track_id: b for b in behaviors}

            # Step 4: Visualization
            draw_tracks(
                frame.image,
                tracks,
                behaviors=behavior_map,
                thickness=config.visualization.bbox_thickness,
                font_scale=config.visualization.font_scale
            )

            # Draw FPS
            if config.visualization.show_fps:
                draw_fps(frame.image, fps)

            # Draw info panel
            info_lines = [
                f"Frame: {frame.frame_number} | Tracks: {len(tracks)} | Detections: {len(detections)}",
                f"Inference: {detector.get_last_inference_time():.1f}ms"
            ]
            y_pos = frame.image.shape[0] - 40
            for line in info_lines:
                cv2.putText(
                    frame.image, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )
                y_pos += 20

            # Draw behavior summary
            draw_behavior_summary(frame.image, behaviors, position=(10, 60))

            # Display
            if not args.no_display:
                cv2.imshow("Week 2: Tracking + Behavior", frame.image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                if key == ord('p'):
                    paused = not paused
                if key == ord('r'):
                    tracker.reset()
                    behavior_analyzer.reset()
                    print("Tracker reset")

            # Print stats every 30 frames
            if frame.frame_number % 30 == 0:
                beh_str = ", ".join([f"{b.track_id}:{b.behavior}" for b in behaviors]) if behaviors else "none"
                print(f"Frame {frame.frame_number}: {len(tracks)} tracks, "
                      f"FPS: {fps:.1f}, Behaviors: {beh_str}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        input_module.close()
        cv2.destroyAllWindows()
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
