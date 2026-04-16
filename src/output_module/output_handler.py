"""
Output Handler - Main module for visualization, logging, and alerting.

Coordinates visualizer, logger, and alert generation.
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import threading

from .visualizer import Visualizer
from .logger import EventLogger


@dataclass
class Alert:
    """
    Alert notification.

    Attributes:
        alert_type: Type of alert (FALLING_DETECTED, etc.)
        track_id: Track that triggered the alert
        severity: low, medium, or high
        message: Human-readable message
        timestamp: When the alert was triggered
    """
    alert_type: str
    track_id: int
    severity: str
    message: str
    timestamp: float


class OutputModule:
    """
    Main output module for the surveillance system.

    Handles visualization, event logging, and alert generation.

    Usage:
        output = OutputModule()

        for frame in video:
            output_frame = output.process(
                frame, tracks, behaviors
            )
            # Display output_frame
    """

    # Alert configuration
    ALERT_BEHAVIORS = {
        "falling": ("FALLING_DETECTED", "high"),
        "fallen": ("PERSON_FALLEN", "high"),
        "running": ("RUNNING_DETECTED", "low"),
    }

    def __init__(
        self,
        log_dir: str = "logs",
        enable_display: bool = True,
        enable_logging: bool = True,
        alert_cooldown_sec: float = 3.0
    ):
        """
        Initialize output module.

        Args:
            log_dir: Directory for log files
            enable_display: Enable visualization
            enable_logging: Enable JSON logging
            alert_cooldown_sec: Minimum seconds between same-type alerts
        """
        self.enable_display = enable_display
        self.enable_logging = enable_logging
        self.alert_cooldown = alert_cooldown_sec

        # Initialize components
        if self.enable_display:
            self.visualizer = Visualizer()
        else:
            self.visualizer = None

        if self.enable_logging:
            self.logger = EventLogger(log_dir=log_dir)
        else:
            self.logger = None

        # FPS calculation
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps_lock = threading.Lock()

        # Alert tracking (for cooldown)
        self._recent_alerts: Dict[str, float] = {}  # alert_type -> timestamp
        self._active_alerts: List[Alert] = []
        self._alert_lock = threading.Lock()

        # Current tracks and stats
        self._num_tracks = 0
        self._num_alerts = 0

    def _update_fps(self):
        """Calculate current FPS."""
        with self._fps_lock:
            self._frame_count += 1
            elapsed = time.time() - self._fps_time

            if elapsed >= 1.0:  # Update every second
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = time.time()

    def get_fps(self) -> float:
        """Get current FPS."""
        with self._fps_lock:
            return self._fps

    def _check_alert(self, behavior_result) -> Optional[Alert]:
        """
        Check if behavior should trigger an alert.

        Args:
            behavior_result: BehaviorResult object

        Returns:
            Alert object if triggered, None otherwise
        """
        behavior = behavior_result.behavior.lower()

        if behavior not in self.ALERT_BEHAVIORS:
            return None

        alert_type, severity = self.ALERT_BEHAVIORS[behavior]

        # Check cooldown
        now = time.time()
        with self._alert_lock:
            last_time = self._recent_alerts.get(alert_type, 0)
            if now - last_time < self.alert_cooldown:
                return None  # Still in cooldown

            # Create alert
            self._recent_alerts[alert_type] = now
            alert = Alert(
                alert_type=alert_type,
                track_id=behavior_result.track_id,
                severity=severity,
                message=f"{alert_type}: Person {behavior_result.track_id}",
                timestamp=now
            )
            self._active_alerts.append(alert)
            return alert

    def _cleanup_old_alerts(self):
        """Remove old alerts from active list."""
        now = time.time()
        with self._alert_lock:
            # Keep alerts from last 10 seconds
            self._active_alerts = [
                a for a in self._active_alerts
                if now - a.timestamp < 10.0
            ]

    def process(
        self,
        frame,
        tracks: List,
        behaviors: List
    ):
        """
        Process frame with tracks and behaviors.

        Args:
            frame: Video frame
            tracks: List of Track objects
            behaviors: List of BehaviorResult objects

        Returns:
            Processed frame (with drawings if enabled)
        """
        # Update FPS
        self._update_fps()

        # Update stats
        self._num_tracks = len(tracks)

        # Log behaviors and check for alerts
        new_alerts = []
        for behavior in behaviors:
            # Log to JSON
            if self.logger:
                self.logger.log_behavior(
                    track_id=behavior.track_id,
                    behavior=behavior.behavior,
                    confidence=behavior.confidence,
                    velocity=behavior.velocity
                )

            # Check for alerts
            alert = self._check_alert(behavior)
            if alert:
                new_alerts.append(alert)
                # Log alert
                if self.logger:
                    self.logger.log_alert(
                        alert_type=alert.alert_type,
                        track_id=alert.track_id,
                        severity=alert.severity,
                        details={
                            "behavior": behavior.behavior,
                            "confidence": behavior.confidence
                        }
                    )

        # Update alert count
        self._cleanup_old_alerts()
        self._num_alerts = len(self._active_alerts)

        # Draw on frame if enabled
        if self.enable_display and self.visualizer:
            # Draw tracks and behaviors
            frame = self.visualizer.draw_tracks(frame, tracks, behaviors)

            # Draw stats
            frame = self.visualizer.draw_stats(
                frame,
                fps=self._fps,
                num_tracks=self._num_tracks,
                num_alerts=self._num_alerts
            )

            # Draw most recent alert
            if self._active_alerts:
                most_recent = max(self._active_alerts, key=lambda a: a.timestamp)
                frame = self.visualizer.draw_alert_overlay(
                    frame,
                    most_recent.message
                )

        return frame

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.

        Returns:
            Dictionary with fps, tracks, alerts
        """
        return {
            "fps": self.get_fps(),
            "active_tracks": self._num_tracks,
            "active_alerts": self._num_alerts,
            "total_alerts": self.logger.event_counts["alert"] if self.logger else 0,
        }

    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """
        Get recent alert events.

        Args:
            count: Number of alerts to return

        Returns:
            List of alert dictionaries
        """
        if self.logger:
            return self.logger.get_recent_alerts(count=count)
        return []

    def reset(self):
        """Reset output module state."""
        with self._fps_lock:
            self._fps = 0.0
            self._frame_count = 0
            self._fps_time = time.time()

        with self._alert_lock:
            self._recent_alerts.clear()
            self._active_alerts.clear()

        self._num_tracks = 0
        self._num_alerts = 0

        if self.logger:
            self.logger.event_counts = {
                "behavior": 0,
                "alert": 0,
                "track": 0,
            }


if __name__ == "__main__":
    """Test output module."""
    import numpy as np
    from src.tracking_module.tracker import Track
    from src.behavior_module.behavior_analyzer import BehaviorResult
    import shutil

    print("Testing OutputModule...")

    # Clean up test logs
    if Path("test_logs").exists():
        shutil.rmtree("test_logs")

    # Create output module
    output = OutputModule(log_dir="test_logs")

    # Create dummy data
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
            bbox=(300, 300, 350, 350),
            confidence=0.85,
            history=[(325, 325) for _ in range(10)],
            age=10
        )
    ]

    behaviors = [
        BehaviorResult(track_id=1, behavior="walking", confidence=0.9, velocity=5.0, aspect_ratio=2.0),
        BehaviorResult(track_id=2, behavior="falling", confidence=0.85, velocity=15.0, aspect_ratio=0.8),
    ]

    # Process frames
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output_frame = output.process(frame, tracks, behaviors)
        time.sleep(0.033)  # ~30fps

    # Get metrics
    metrics = output.get_metrics()
    print(f"Metrics: {metrics}")

    # Get recent alerts
    alerts = output.get_recent_alerts(count=5)
    print(f"Recent alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['alert_type']} (ID:{alert['track_id']})")

    print("\nOutputModule test complete!")

    # Cleanup
    shutil.rmtree("test_logs")
