"""
Optimized Real-Time Video Detection System

Achieves 20-30+ FPS on CPU through:
    1. Frame skipping (process every Nth frame)
    2. Reduced input resolution (640x480)
    3. Lightweight YOLOv8n model
    4. Higher confidence threshold (0.6)
    5. Simplified tracking without appearance features
    6. Minimized drawing overhead
    7. Efficient data structures

Usage:
    python app/main_optimized.py
    python app/main_optimized.py --source data/samples/test_video.mp4

Controls:
    q - quit
    p - pause/resume
    r - reset tracker
    s - toggle stats display
"""

import sys
import cv2
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_module.input_handler import InputModule
from src.detection_module.detector import DetectionModule, DetectionResult
from src.tracking_module.tracker import TrackingModule
from src.behavior_module.behavior_analyzer import BehaviorAnalyzer
from src.utils.config_loader import load_config


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

class PerformanceConfig:
    """Centralized performance tuning parameters."""

    # Frame processing
    PROCESS_EVERY_N_FRAMES = 2  # Run detection every 2nd frame (was: every frame)
                              # Higher = faster FPS but less responsive tracking

    # Input resolution (resize for speed)
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 480

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.6  # Higher threshold = fewer detections = faster (was: 0.5)
    PERSON_CLASS_ONLY = [0]     # Class 0 = person only

    # Tracking settings - optimized for speed
    TRACKER_MAX_AGE = 15        # Reduced from 30 (fewer frames to track)
    TRACKER_MIN_HITS = 2        # Reduced from 3 (faster track confirmation)
    TRACKER_IOU_THRESHOLD = 0.4  # Increased from 0.3 (stricter matching = fewer updates)

    # Visualization settings
    SHOW_FPS = True
    SHOW_DETECTION_COUNT = True
    SHOW_TRACK_COUNT = True
    DRAW_TRAJECTORIES = False   # Disable expensive trajectory drawing (was: True)
    FONT_SCALE = 0.5            # Reduced from 0.6
    BBOX_THICKNESS = 2

    # Model settings
    MODEL_PATH = "yolov8n.pt"   # YOLOv8 nano - smallest and fastest

    # Display settings
    DISPLAY_SCALE = 1.0         # Scale factor for display (can reduce to 0.75 for speed)


# ============================================================================
# OPTIMIZED VISUALIZATION
# ============================================================================

class OptimizedVisualizer:
    """Minimal overhead visualization for maximum FPS."""

    # Color cache (BGR format)
    COLORS = {
        "person": (0, 255, 0),      # Green
        "walking": (0, 255, 0),     # Green
        "running": (0, 0, 255),     # Red
        "falling": (0, 0, 255),     # Red
        "fallen": (0, 0, 255),      # Red
        "standing": (255, 255, 0),  # Cyan
        "unknown": (128, 128, 128), # Gray
    }

    def __init__(self, config: PerformanceConfig):
        self.config = config
        # Pre-allocate text cache for common labels
        self._text_cache: Dict[str, Tuple] = {}

    def _get_text_size(self, text: str) -> Tuple[int, int]:
        """Cache text size calculations."""
        if text not in self._text_cache:
            (w, h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 1
            )
            self._text_cache[text] = (w, h)
        return self._text_cache[text]

    def draw_track(self, frame: np.ndarray, track, behavior=None) -> None:
        """Draw a single track with minimal overhead."""
        x1, y1, x2, y2 = map(int, track.bbox)

        # Get color based on behavior
        color = self.COLORS.get(
            behavior.behavior if behavior else "person",
            self.COLORS["unknown"]
        )

        # Draw bounding box (simple rectangle, no fancy effects)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.BBOX_THICKNESS)

        # Build minimal label: "ID:behavior"
        label = f"{track.track_id}"
        if behavior:
            label += f":{behavior.behavior[:3]}"  # Shortened behavior name

        # Calculate text size
        text_w, text_h = self._get_text_size(label)

        # Draw label background
        label_y = y1 - 5 if y1 > 25 else y2 + text_h + 5
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 2),
            (x1 + text_w + 4, label_y + 2),
            color,
            -1
        )

        # Draw text (white on colored background)
        cv2.putText(
            frame, label, (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    def draw_tracks(self, frame: np.ndarray, tracks, behaviors=None) -> None:
        """Draw all tracks efficiently."""
        behaviors = behaviors or {}
        for track in tracks:
            behavior = behaviors.get(track.track_id)
            self.draw_track(frame, track, behavior)

    def draw_stats(self, frame: np.ndarray, fps: float, num_detections: int,
                   num_tracks: int, inference_time_ms: float) -> None:
        """Draw performance stats in corner."""
        stats_text = f"FPS:{fps:.1f}|D:{num_detections}|T:{num_tracks}|{inference_time_ms:.1f}ms"

        # Draw semi-transparent background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame, stats_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )


# ============================================================================
# FRAME SKIP MANAGER
# ============================================================================

class FrameSkipManager:
    """
    Manages intelligent frame skipping for optimal performance.

    Instead of running detection on every frame, we:
    1. Run detection every N frames
    2. Use tracker predictions for intermediate frames
    3. Adjust skip rate based on scene complexity (optional)
    """

    def __init__(self, skip_interval: int = 2):
        self.skip_interval = skip_interval
        self.frame_count = 0
        self.last_detections: List[DetectionResult] = []
        self.last_tracks = []

    def should_detect(self) -> bool:
        """Returns True if detection should run this frame."""
        return self.frame_count % self.skip_interval == 0

    def increment(self) -> None:
        """Increment frame counter."""
        self.frame_count += 1

    def update_detections(self, detections: List[DetectionResult]) -> None:
        """Cache last detections for interpolation."""
        self.last_detections = detections


# ============================================================================
# OPTIMIZED PIPELINE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized Real-Time Detection Pipeline (20-30+ FPS)"
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
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Process detection every N frames (default: 2, higher = faster)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Input resolution WxH (default: 640x480, use 320x240 for max speed)"
    )
    return parser.parse_args()


def main():
    """Optimized main pipeline with frame skipping and reduced overhead."""
    args = parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        width, height = 640, 480

    # Initialize performance config
    config = PerformanceConfig()
    config.PROCESS_EVERY_N_FRAMES = args.skip_frames
    config.INPUT_WIDTH = width
    config.INPUT_HEIGHT = height

    print("=" * 60)
    print("OPTIMIZED Real-Time Detection Pipeline")
    print("=" * 60)
    print(f"\nPerformance Settings:")
    print(f"  Resolution: {config.INPUT_WIDTH}x{config.INPUT_HEIGHT}")
    print(f"  Detection every: {config.PROCESS_EVERY_N_FRAMES} frames")
    print(f"  Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  Trajectory drawing: {config.DRAW_TRAJECTORIES}")

    # Load base config
    base_config = load_config(args.config)

    # Determine video source
    source = args.source if args.source is not None else base_config.input.source
    if source == "0" or source == 0:
        source = 0
        print("\nSource: Webcam (0)")
    else:
        print(f"\nSource: {source}")

    # Initialize modules
    print("\n[1/5] Initializing InputModule...")
    input_module = InputModule(
        source=source,
        width=config.INPUT_WIDTH,
        height=config.INPUT_HEIGHT,
        fps=30
    )

    print("\n[2/5] Initializing DetectionModule (YOLOv8n)...")
    detector = DetectionModule(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        classes=config.PERSON_CLASS_ONLY,
        device="cpu"  # Force CPU for consistency
    )
    detector.warmup(image_size=(config.INPUT_WIDTH, config.INPUT_HEIGHT))

    print("\n[3/5] Initializing TrackingModule...")
    tracker = TrackingModule(
        max_age=config.TRACKER_MAX_AGE,
        min_hits=config.TRACKER_MIN_HITS,
        iou_threshold=config.TRACKER_IOU_THRESHOLD,
        history_size=15  # Reduced from 30
    )

    print("\n[4/5] Initializing BehaviorModule...")
    behavior_analyzer = BehaviorAnalyzer(
        walking_velocity_range=(2.0, 8.0),
        running_velocity_threshold=8.0,
        falling_aspect_ratio_threshold=1.0,
        fps=30.0 / config.PROCESS_EVERY_N_FRAMES  # Adjust for frame skip
    )

    print("\n[5/5] Initializing Visualizer...")
    visualizer = OptimizedVisualizer(config)

    # Frame skip manager
    skip_manager = FrameSkipManager(skip_interval=config.PROCESS_EVERY_N_FRAMES)

    # FPS calculation - use deque for efficient moving average
    fps_deque: deque = deque(maxlen=30)
    last_time = time.time()

    print("\n" + "=" * 60)
    print("Pipeline ready! Target: 20-30+ FPS")
    print("Controls: q=quit | p=pause | r=reset | s=stats toggle")
    print("=" * 60 + "\n")

    paused = False
    show_stats = True
    stats_counter = 0

    try:
        for frame in input_module.stream():
            current_time = time.time()

            # Calculate FPS
            fps_deque.append(current_time)
            fps = len(fps_deque) / (fps_deque[-1] - fps_deque[0]) if len(fps_deque) > 1 else 0

            # Handle pause
            if paused:
                cv2.imshow("Optimized Detection", frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = False
                    print("Resumed")
                continue

            # === OPTIMIZED PIPELINE ===

            # Step 1: Decide if we run detection this frame
            run_detection = skip_manager.should_detect()
            skip_manager.increment()

            if run_detection:
                # Run full pipeline: Detection -> Tracking -> Behavior
                detections = detector.detect(frame.image)
                skip_manager.update_detections(detections)

                # Update tracker with detections
                tracks = tracker.update(detections, frame=frame.image)

                # Analyze behaviors
                behaviors_list = behavior_analyzer.analyze(tracks)
                behaviors = {b.track_id: b for b in behaviors_list}

                # Cache for skipped frames
                skip_manager.last_tracks = tracks
            else:
                # Skip detection: just update tracker with empty detections
                # (tracker will use motion prediction)
                tracks = tracker.update([], frame=frame.image)

                # Use cached behaviors or re-analyze with predicted positions
                if skip_manager.last_tracks:
                    behaviors_list = behavior_analyzer.analyze(tracks)
                    behaviors = {b.track_id: b for b in behaviors_list}
                else:
                    behaviors = {}

            # Step 2: Visualization (only if display enabled)
            if not args.no_display:
                visualizer.draw_tracks(frame.image, tracks, behaviors)

                if show_stats:
                    visualizer.draw_stats(
                        frame.image, fps,
                        len(skip_manager.last_detections) if run_detection else 0,
                        len(tracks),
                        detector.get_last_inference_time()
                    )

                # Display
                cv2.imshow("Optimized Detection", frame.image)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = True
                    print("Paused. Press 'p' to resume.")
                elif key == ord('r'):
                    tracker.reset()
                    behavior_analyzer.reset()
                    skip_manager.last_detections = []
                    skip_manager.last_tracks = []
                    print("Tracker reset")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Stats display: {'ON' if show_stats else 'OFF'}")

            # Print stats every 60 frames (every ~2 seconds at 30fps)
            stats_counter += 1
            if stats_counter % 60 == 0:
                beh_str = ", ".join([f"{b.track_id}:{b.behavior[:3]}" for b in behaviors.values()]) if behaviors else "none"
                print(f"Frame {frame.frame_number}: {len(tracks)} tracks, "
                      f"FPS: {fps:.1f}, Inference: {detector.get_last_inference_time():.1f}ms, "
                      f"Behaviors: {beh_str}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        input_module.close()
        cv2.destroyAllWindows()
        print("\nPipeline stopped.")
        print(f"\nFinal Stats:")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Frames processed: {frame.frame_number}")


if __name__ == "__main__":
    main()
