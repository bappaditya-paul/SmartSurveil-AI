"""
Tracking Module

DeepSORT wrapper for multi-person tracking.
Assigns persistent track IDs and maintains position history.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class Track:
    """
    Single track result with history.

    Attributes:
        track_id: Unique persistent ID (1, 2, 3...)
        bbox: Current bounding box (x1, y1, x2, y2)
        confidence: Detection confidence
        history: List of past center positions [(x, y), ...]
        age: Number of frames this track has existed
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    history: List[Tuple[float, float]]
    age: int


class TrackingModule:
    """
    DeepSORT-based multi-person tracker.

    Usage:
        tracker = TrackingModule(max_age=30)

        for frame in video:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            for track in tracks:
                print(f"Person {track.track_id} at {track.bbox}")
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        history_size: int = 30
    ):
        """
        Initialize DeepSORT tracker.

        Args:
            max_age: Delete track if no match for this many frames
            min_hits: Confirm track only after this many detections
            iou_threshold: IoU threshold for matching
            history_size: Number of past positions to keep per track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.history_size = history_size

        # Initialize DeepSORT
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=min_hits,
                max_iou_distance=1 - iou_threshold,  # convert IoU to distance
                embedder='mobilenet',  # lightweight ReID model
                half=True,  # use half precision (faster)
                bgr=True  # OpenCV uses BGR
            )
        except ImportError:
            raise ImportError(
                "deep_sort_realtime not installed. "
                "Run: pip install deep-sort-realtime"
            )

        self.frame_count = 0
        self._track_histories = defaultdict(list)  # track_id -> positions

    def _convert_detections(self, detections):
        """
        Convert our DetectionResult format to DeepSORT format.

        DeepSORT expects: [[x1, y1, w, h], conf, class_name]
        """
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            ds_detections.append([[x1, y1, w, h], det.confidence, det.class_name])
        return ds_detections

    def update(self, detections, frame=None) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of DetectionResult from DetectionModule
            frame: Optional image for appearance features

        Returns:
            List of confirmed Track objects
        """
        self.frame_count += 1

        # Convert to DeepSORT format
        ds_dets = self._convert_detections(detections)

        # Update tracker
        if frame is not None:
            # Ensure frame is uint8 (required by DeepSORT embedder)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            tracks_raw = self.tracker.update_tracks(ds_dets, frame=frame)
        else:
            # Create dummy frame if not provided (appearance features disabled)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            tracks_raw = self.tracker.update_tracks(ds_dets, frame=dummy_frame)

        # Convert to our Track format
        tracks = []
        for trk in tracks_raw:
            if not trk.is_confirmed():
                continue  # Skip tentative tracks

            track_id = trk.track_id
            x1, y1, x2, y2 = map(int, trk.to_tlbr())  # top-left, bottom-right

            # Calculate center point for history
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Update history
            self._track_histories[track_id].append((cx, cy))
            if len(self._track_histories[track_id]) > self.history_size:
                self._track_histories[track_id].pop(0)

            # Create Track object
            track = Track(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=trk.get_det_conf() or 0.5,
                history=self._track_histories[track_id].copy(),
                age=trk.age if hasattr(trk, 'age') else self.frame_count
            )
            tracks.append(track)

        # Cleanup old histories for deleted tracks
        active_ids = {t.track_id for t in tracks}
        for tid in list(self._track_histories.keys()):
            if tid not in active_ids:
                del self._track_histories[tid]

        return tracks

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Get position history for a specific track."""
        return self._track_histories.get(track_id, [])

    def reset(self):
        """Reset all tracks."""
        self.frame_count = 0
        self._track_histories.clear()
        # Reinitialize tracker
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                max_iou_distance=1 - self.iou_threshold,
                embedder='mobilenet',
                half=True,
                bgr=True
            )
        except ImportError:
            pass


if __name__ == "__main__":
    """Test tracking with synthetic data."""
    from src.detection_module.detector import DetectionResult

    print("Testing TrackingModule...")

    tracker = TrackingModule()

    # Simulate detections moving across frames
    for frame_idx in range(10):
        # Two persons moving
        detections = [
            DetectionResult(
                bbox=(100 + frame_idx * 5, 100, 150 + frame_idx * 5, 200),
                confidence=0.9,
                class_id=0,
                class_name="person"
            ),
            DetectionResult(
                bbox=(300, 100 + frame_idx * 3, 350, 200 + frame_idx * 3),
                confidence=0.85,
                class_id=0,
                class_name="person"
            )
        ]

        tracks = tracker.update(detections)
        print(f"Frame {frame_idx}: {len(tracks)} tracks")
        for t in tracks:
            print(f"  ID {t.track_id}: {t.bbox}, history: {len(t.history)} points")

    print("\nTracking test complete!")
