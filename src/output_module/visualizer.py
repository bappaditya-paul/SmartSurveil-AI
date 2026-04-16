"""
Visualizer - Draw bounding boxes, IDs, and behaviors on frames.

Simple drawing utilities for visualization.
"""

import cv2
from typing import List, Tuple, Optional


class Visualizer:
    """
    Draw detections, tracks, and behaviors on video frames.

    Usage:
        viz = Visualizer()
        frame = viz.draw_tracks(frame, tracks, behaviors)
    """

    # Color scheme for different behaviors
    COLORS = {
        "walking": (0, 255, 0),      # Green
        "running": (0, 165, 255),    # Orange
        "falling": (0, 0, 255),      # Red
        "fallen": (0, 0, 200),       # Dark Red
        "standing": (255, 255, 0),    # Cyan
        "unknown": (128, 128, 128),  # Gray
    }

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ):
        """
        Initialize visualizer.

        Args:
            box_thickness: Thickness of bounding box lines
            font_scale: Size of text labels
            font_thickness: Thickness of text
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def _get_color(self, behavior: str) -> Tuple[int, int, int]:
        """Get color based on behavior type."""
        return self.COLORS.get(behavior.lower(), self.COLORS["unknown"])

    def draw_box(
        self,
        frame,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        label: Optional[str] = None
    ):
        """
        Draw a bounding box with optional label.

        Args:
            frame: Image to draw on
            bbox: (x1, y1, x2, y2) coordinates
            color: BGR color tuple
            label: Text label to draw above box
        """
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self.box_thickness
        )

        # Draw label if provided
        if label:
            # Calculate text size for background
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )

    def draw_tracks(
        self,
        frame,
        tracks: List,
        behaviors: List,
        show_trail: bool = True
    ):
        """
        Draw all tracks with their behaviors.

        Args:
            frame: Image to draw on
            tracks: List of Track objects
            behaviors: List of BehaviorResult objects
            show_trail: Whether to draw movement trail

        Returns:
            Frame with drawings
        """
        # Create behavior lookup by track_id
        behavior_map = {b.track_id: b for b in behaviors}

        for track in tracks:
            # Get behavior for this track
            behavior = behavior_map.get(track.track_id)

            if behavior:
                color = self._get_color(behavior.behavior)
                label = f"ID:{track.track_id} {behavior.behavior}"
            else:
                color = self.COLORS["unknown"]
                label = f"ID:{track.track_id}"

            # Draw bounding box
            self.draw_box(frame, track.bbox, color, label)

            # Draw movement trail
            if show_trail and len(track.history) > 1:
                points = [(int(p[0]), int(p[1])) for p in track.history]
                for i in range(1, len(points)):
                    # Fade out older points
                    alpha = i / len(points)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, thickness)

        return frame

    def draw_stats(
        self,
        frame,
        fps: float,
        num_tracks: int,
        num_alerts: int
    ):
        """
        Draw system stats on frame.

        Args:
            frame: Image to draw on
            fps: Current FPS
            num_tracks: Number of active tracks
            num_alerts: Number of alerts

        Returns:
            Frame with stats overlay
        """
        # Background for stats
        cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 100), (255, 255, 255), 1)

        # Draw stats
        stats = [
            f"FPS: {fps:.1f}",
            f"Tracks: {num_tracks}",
            f"Alerts: {num_alerts}",
        ]

        y_offset = 35
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25

        return frame

    def draw_alert_overlay(
        self,
        frame,
        alert_message: str,
        duration_ms: int = 2000
    ):
        """
        Draw a prominent alert overlay on the frame.

        Args:
            frame: Image to draw on
            alert_message: Alert text to display
            duration_ms: How long alert should be shown (not used here)

        Returns:
            Frame with alert
        """
        h, w = frame.shape[:2]

        # Draw red banner at top
        banner_height = 60
        cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 255), -1)
        cv2.rectangle(frame, (0, 0), (w, banner_height), (255, 255, 255), 2)

        # Draw alert text
        (text_w, text_h), _ = cv2.getTextSize(
            alert_message,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            3
        )

        text_x = (w - text_w) // 2
        text_y = (banner_height + text_h) // 2

        cv2.putText(
            frame,
            alert_message,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        return frame


if __name__ == "__main__":
    """Test visualizer with dummy data."""
    import numpy as np
    from src.tracking_module.tracker import Track
    from src.behavior_module.behavior_analyzer import BehaviorResult

    print("Testing Visualizer...")

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create dummy tracks
    tracks = [
        Track(
            track_id=1,
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            history=[(150 + i*2, 200) for i in range(10)],
            age=10
        ),
        Track(
            track_id=2,
            bbox=(300, 150, 400, 350),
            confidence=0.85,
            history=[(350, 250 - i*3) for i in range(10)],
            age=10
        )
    ]

    # Create dummy behaviors
    behaviors = [
        BehaviorResult(track_id=1, behavior="walking", confidence=0.9, velocity=5.0, aspect_ratio=2.0),
        BehaviorResult(track_id=2, behavior="falling", confidence=0.8, velocity=10.0, aspect_ratio=0.8),
    ]

    # Draw
    viz = Visualizer()
    frame = viz.draw_tracks(frame, tracks, behaviors)
    frame = viz.draw_stats(frame, fps=25.5, num_tracks=2, num_alerts=1)
    frame = viz.draw_alert_overlay(frame, "ALERT: Person Falling Detected!")

    # Save test image
    cv2.imwrite("test_visualization.jpg", frame)
    print("Saved test visualization to test_visualization.jpg")
