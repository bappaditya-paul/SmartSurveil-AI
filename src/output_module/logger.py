"""
Event Logger - JSON logging for surveillance events.

Logs events in JSON format for easy parsing and analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import threading


class EventLogger:
    """
    JSON event logger for surveillance system.

    Logs detections, behaviors, and alerts to JSON files.

    Usage:
        logger = EventLogger(log_dir="logs")
        logger.log_behavior(track_id=1, behavior="falling", confidence=0.9)
        logger.log_alert("FALLING_DETECTED", track_id=1)
    """

    def __init__(
        self,
        log_dir: str = "logs",
        max_file_size_mb: float = 10.0,
        max_files: int = 5
    ):
        """
        Initialize event logger.

        Args:
            log_dir: Directory to store log files
            max_file_size_mb: Rotate file when it reaches this size
            max_files: Number of rotated files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_files = max_files

        # Current log file
        self.current_log_file = self._get_log_filename()
        self._lock = threading.Lock()

        # Event counters for stats
        self.event_counts = {
            "behavior": 0,
            "alert": 0,
            "track": 0,
        }

    def _get_log_filename(self) -> Path:
        """Generate log filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"events_{timestamp}.jsonl"

    def _check_rotation(self):
        """Rotate log file if it exceeds max size."""
        if self.current_log_file.exists():
            if self.current_log_file.stat().st_size > self.max_file_size:
                self.current_log_file = self._get_log_filename()
                self._cleanup_old_files()

    def _cleanup_old_files(self):
        """Remove old log files if exceeding max_files."""
        log_files = sorted(
            self.log_dir.glob("events_*.jsonl"),
            key=lambda p: p.stat().st_mtime
        )
        while len(log_files) >= self.max_files:
            log_files[0].unlink()
            log_files.pop(0)

    def _write_event(self, event: Dict[str, Any]):
        """Write event to log file."""
        with self._lock:
            self._check_rotation()

            event["timestamp"] = datetime.now().isoformat()
            event["epoch_time"] = time.time()

            with open(self.current_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_behavior(
        self,
        track_id: int,
        behavior: str,
        confidence: float,
        velocity: float = 0.0,
        bbox: Optional[tuple] = None
    ):
        """
        Log a behavior detection event.

        Args:
            track_id: Track ID
            behavior: Detected behavior (walking, running, falling, etc.)
            confidence: Detection confidence
            velocity: Current velocity
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        event = {
            "event_type": "behavior",
            "track_id": track_id,
            "behavior": behavior,
            "confidence": confidence,
            "velocity": velocity,
        }
        if bbox:
            event["bbox"] = bbox

        self._write_event(event)
        self.event_counts["behavior"] += 1

    def log_alert(
        self,
        alert_type: str,
        track_id: int,
        severity: str = "medium",
        details: Optional[Dict] = None
    ):
        """
        Log an alert event.

        Args:
            alert_type: Type of alert (FALLING_DETECTED, etc.)
            track_id: Track ID that triggered alert
            severity: Alert severity (low, medium, high)
            details: Additional alert details
        """
        event = {
            "event_type": "alert",
            "alert_type": alert_type,
            "track_id": track_id,
            "severity": severity,
        }
        if details:
            event["details"] = details

        self._write_event(event)
        self.event_counts["alert"] += 1

    def log_track(self, track_id: int, bbox: tuple, confidence: float):
        """
        Log a new track creation.

        Args:
            track_id: New track ID
            bbox: Initial bounding box
            confidence: Detection confidence
        """
        event = {
            "event_type": "track",
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
        }
        self._write_event(event)
        self.event_counts["track"] += 1

    def get_recent_alerts(
        self,
        count: int = 10,
        alert_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get recent alert events.

        Args:
            count: Number of alerts to return
            alert_types: Filter by specific alert types

        Returns:
            List of alert events (most recent first)
        """
        alerts = []

        # Read from all log files
        log_files = sorted(
            self.log_dir.glob("events_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for log_file in log_files:
            if len(alerts) >= count:
                break

            with open(log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("event_type") == "alert":
                            if alert_types is None or event.get("alert_type") in alert_types:
                                alerts.append(event)
                                if len(alerts) >= count:
                                    break
                    except json.JSONDecodeError:
                        continue

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_events": sum(self.event_counts.values()),
            "behavior_events": self.event_counts["behavior"],
            "alert_events": self.event_counts["alert"],
            "track_events": self.event_counts["track"],
            "log_dir": str(self.log_dir),
            "current_file": str(self.current_log_file),
        }


if __name__ == "__main__":
    """Test event logger."""
    import shutil

    print("Testing EventLogger...")

    # Clean up test dir
    test_dir = "test_logs"
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)

    logger = EventLogger(log_dir=test_dir)

    # Log some events
    logger.log_behavior(1, "walking", 0.9, 5.0, (100, 100, 200, 300))
    logger.log_behavior(2, "falling", 0.85, 15.0, (300, 300, 350, 350))
    logger.log_alert("FALLING_DETECTED", 2, "high", {"velocity": 15.0})
    logger.log_track(1, (100, 100, 200, 300), 0.9)

    # Get recent alerts
    alerts = logger.get_recent_alerts(count=5)
    print(f"Recent alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['alert_type']} (ID:{alert['track_id']})")

    # Get stats
    stats = logger.get_stats()
    print(f"\nStats: {stats}")

    print("\nEventLogger test complete!")

    # Cleanup
    shutil.rmtree(test_dir)
