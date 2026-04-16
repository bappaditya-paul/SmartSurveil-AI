"""
Visualization Utilities

Functions to draw bounding boxes, tracks, trajectories, and behavior labels.
"""

import cv2
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Simple detection result container."""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    class_name: str = "person"


# Behavior color mapping (BGR format)
BEHAVIOR_COLORS = {
    "walking": (0, 255, 0),      # Green
    "running": (0, 0, 255),      # Red
    "falling": (0, 0, 255),      # Red (flashing handled separately)
    "fallen": (0, 0, 255),       # Red
    "standing": (255, 255, 0),   # Cyan
    "unknown": (128, 128, 128),  # Gray
}


def draw_detection(
    frame,
    detection: Detection,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6
):
    """
    Draw a single bounding box with label.

    Args:
        frame: OpenCV image (numpy array)
        detection: Detection object with bbox and confidence
        color: BGR color tuple (default green)
        thickness: Line thickness
        font_scale: Text size
    """
    x1, y1, x2, y2 = map(int, detection.bbox)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Create label text
    label = f"{detection.class_name}: {detection.confidence:.2f}"

    # Calculate text size for background
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )

    # Draw filled rectangle behind text for readability
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - 10),
        (x1 + text_w, y1),
        color,
        -1  # Filled
    )

    # Draw text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White text
        1,
        cv2.LINE_AA
    )


def draw_detections(
    frame,
    detections: List[Detection],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6
):
    """Draw multiple detections on frame."""
    for det in detections:
        draw_detection(frame, det, color, thickness, font_scale)


def draw_track(
    frame,
    track_id: int,
    bbox: tuple,
    behavior: Optional[str] = None,
    confidence: float = 0.0,
    history: Optional[List[Tuple[float, float]]] = None,
    color: Optional[tuple] = None,
    thickness: int = 2,
    font_scale: float = 0.6
):
    """
    Draw a track with ID, optional behavior, and trajectory.

    Args:
        frame: OpenCV image
        track_id: Person ID
        bbox: (x1, y1, x2, y2)
        behavior: Classified behavior (walking, running, falling, etc.)
        confidence: Behavior confidence
        history: List of past center positions for trajectory
        color: Box color (None = use behavior color)
        thickness: Line thickness
        font_scale: Text size
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Determine color based on behavior
    if color is None:
        color = BEHAVIOR_COLORS.get(behavior or "unknown", (128, 128, 128))

    # Draw trajectory if history exists
    if history and len(history) >= 2:
        points = [(int(p[0]), int(p[1])) for p in history]
        for i in range(1, len(points)):
            # Fade color from old to new
            alpha = i / len(points)
            traj_color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)
            cv2.line(frame, points[i-1], points[i], traj_color, 1)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build label
    label = f"ID:{track_id}"
    if behavior:
        label += f" {behavior}"
        if confidence > 0:
            label += f"({confidence:.2f})"

    # Calculate text size
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )

    # Draw label background
    label_y = y1 - 10 if y1 > 30 else y2 + text_h + 5
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 5),
        (x1 + text_w + 5, label_y + 5),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1 + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def draw_tracks(
    frame,
    tracks,
    behaviors=None,
    thickness: int = 2,
    font_scale: float = 0.6
):
    """
    Draw multiple tracks with behaviors.

    Args:
        frame: OpenCV image
        tracks: List of Track objects
        behaviors: Dict mapping track_id -> BehaviorResult
    """
    behaviors = behaviors or {}

    for track in tracks:
        behavior = behaviors.get(track.track_id)
        if behavior:
            draw_track(
                frame,
                track_id=track.track_id,
                bbox=track.bbox,
                behavior=behavior.behavior,
                confidence=behavior.confidence,
                history=track.history,
                thickness=thickness,
                font_scale=font_scale
            )
        else:
            draw_track(
                frame,
                track_id=track.track_id,
                bbox=track.bbox,
                history=track.history,
                thickness=thickness,
                font_scale=font_scale
            )


def draw_fps(frame, fps: float, font_scale: float = 0.7):
    """Draw FPS counter in top-left corner."""
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),  # Yellow
        2,
        cv2.LINE_AA
    )


def draw_info_panel(frame, text: str, position: tuple = (10, 60)):
    """Draw additional info text on frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


def draw_behavior_summary(frame, behaviors, position: tuple = (10, 90)):
    """Draw summary of all behaviors in scene."""
    if not behaviors:
        return

    y = position[1]
    summary = "Behaviors: "
    counts = {}
    for b in behaviors:
        counts[b.behavior] = counts.get(b.behavior, 0) + 1

    summary += ", ".join([f"{k}:{v}" for k, v in counts.items()])

    cv2.putText(
        frame,
        summary,
        (position[0], y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
