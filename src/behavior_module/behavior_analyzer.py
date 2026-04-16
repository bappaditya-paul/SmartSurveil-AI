"""
Behavior Analyzer

Improved behavior classification with smoothed velocity
and better fall detection using aspect ratio + velocity.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import time


@dataclass
class BehaviorResult:
    """
    Behavior classification result.

    Attributes:
        track_id: Person being analyzed
        behavior: Classified behavior (walking, running, falling, standing)
        confidence: Confidence score (0.0-1.0)
        velocity: Current speed in pixels/second (smoothed)
        aspect_ratio: Current bbox aspect ratio (height/width)
    """
    track_id: int
    behavior: str
    confidence: float
    velocity: float
    aspect_ratio: float


class BehaviorAnalyzer:
    """
    Improved rule-based behavior classifier.

    Features:
    - Smoothed velocity (average over multiple frames)
    - Better fall detection (AR change + vertical velocity)
    - Improved running detection with hysteresis

    Usage:
        analyzer = BehaviorAnalyzer()

        for tracks in tracker.update():
            behaviors = analyzer.analyze(tracks)
            for b in behaviors:
                print(f"Person {b.track_id} is {b.behavior}")
    """

    def __init__(
        self,
        # Velocity smoothing
        velocity_window: int = 5,
        # Walking thresholds
        walking_min: float = 30.0,
        walking_max: float = 120.0,
        # Running threshold (with hysteresis)
        running_threshold: float = 120.0,
        running_min_frames: int = 3,
        # Fall detection
        fall_ar_threshold: float = 0.8,
        fall_ar_change_threshold: float = 1.5,
        fall_vertical_velocity_threshold: float = 80.0,
        # Detection fps (for velocity calculation)
        fps: float = 30.0
    ):
        """
        Initialize behavior analyzer.

        Args:
            velocity_window: Frames to average for smoothing
            walking_min: Min pixels/sec for walking
            walking_max: Max pixels/sec for walking (below = walking, above = running)
            running_threshold: Threshold for running detection
            running_min_frames: Frames of fast movement before calling it "running"
            fall_ar_threshold: AR below this indicates lying down
            fall_ar_change_threshold: AR must drop by this much to trigger fall
            fall_vertical_velocity_threshold: Min downward velocity for fall
            fps: Video frame rate for velocity calculation
        """
        # Store track history for velocity/AR calculation
        # track_id -> deque of (timestamp, position, aspect_ratio, velocity)
        self._track_data = defaultdict(lambda: deque(maxlen=30))

        # Velocity smoothing
        self.velocity_window = velocity_window
        self._velocity_history = defaultdict(lambda: deque(maxlen=velocity_window))

        # Running detection hysteresis
        self.running_threshold = running_threshold
        self.running_min_frames = running_min_frames
        self._fast_frames = defaultdict(int)
        self._was_running = defaultdict(bool)

        # Detection thresholds
        self.walking_min = walking_min
        self.walking_max = walking_max
        self.fall_ar_threshold = fall_ar_threshold
        self.fall_ar_change = fall_ar_change_threshold
        self.fall_velocity_threshold = fall_vertical_velocity_threshold

        # Time calculation
        self.frame_time = 1.0 / fps

    def _get_smoothed_velocity(self, track_id: int, instant_velocity: float) -> float:
        """
        Calculate smoothed velocity over multiple frames.

        Args:
            track_id: Track ID
            instant_velocity: Current frame's instantaneous velocity

        Returns:
            Smoothed velocity (average of last frames)
        """
        self._velocity_history[track_id].append(instant_velocity)
        return np.mean(self._velocity_history[track_id])

    def _calculate_instant_velocity(
        self,
        track_id: int,
        current_pos: Tuple[float, float]
    ) -> float:
        """
        Calculate instantaneous velocity from position history.

        Args:
            track_id: Track ID
            current_pos: (x, y) center position

        Returns:
            Velocity in pixels per second
        """
        history = self._track_data.get(track_id)
        if not history or len(history) < 2:
            return 0.0

        # Get previous position
        prev = history[-1]
        prev_pos = prev[1]

        # Calculate distance
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Velocity = distance / time
        return distance / self.frame_time

    def _calculate_vertical_velocity(
        self,
        track_id: int,
        current_pos: Tuple[float, float]
    ) -> float:
        """
        Calculate vertical (downward) velocity.

        Args:
            track_id: Track ID
            current_pos: (x, y) center position

        Returns:
            Downward velocity in pixels per second
        """
        history = self._track_data.get(track_id)
        if not history or len(history) < 2:
            return 0.0

        prev = history[-1]
        prev_y = prev[1][1]
        current_y = current_pos[1]

        # Positive = moving down (falling)
        return (current_y - prev_y) / self.frame_time

    def _get_previous_aspect_ratio(self, track_id: int) -> Optional[float]:
        """Get previous aspect ratio for change detection."""
        history = self._track_data.get(track_id)
        if not history or len(history) < 2:
            return None
        return history[-1][2]

    def _detect_fall(
        self,
        track_id: int,
        aspect_ratio: float,
        vertical_velocity: float
    ) -> Tuple[bool, float]:
        """
        Detect fall using aspect ratio change + vertical velocity.

        A fall is detected when:
        1. Aspect ratio drops significantly (standing ~2-3 -> fallen ~0.5-0.8)
        2. AND there was recent downward vertical velocity

        Args:
            track_id: Track ID
            aspect_ratio: Current bbox AR
            vertical_velocity: Current vertical velocity

        Returns:
            (is_falling, confidence)
        """
        prev_ar = self._get_previous_aspect_ratio(track_id)
        if prev_ar is None:
            return False, 0.0

        # Check 1: Is person now lying down (low AR)?
        is_lying = aspect_ratio < self.fall_ar_threshold

        # Check 2: Did AR drop significantly?
        ar_drop = prev_ar - aspect_ratio
        significant_drop = ar_drop > self.fall_ar_change

        # Check 3: Was there downward movement?
        moving_down = vertical_velocity > self.fall_velocity_threshold

        # Fall detection: AR dropped significantly AND (now lying OR moving down fast)
        if significant_drop and (is_lying or moving_down):
            # Calculate confidence based on how much AR dropped and vertical velocity
            ar_confidence = min(1.0, ar_drop / 2.0)
            velocity_confidence = min(1.0, vertical_velocity / 150.0)
            confidence = max(ar_confidence, velocity_confidence)
            return True, confidence

        # Check if already fallen (low AR for multiple frames)
        if is_lying and not significant_drop:
            # Person has been down for a while
            return False, 0.7  # Return "fallen" not "falling"

        return False, 0.0

    def _classify_running(
        self,
        track_id: int,
        velocity: float
    ) -> Tuple[bool, float]:
        """
        Classify running with hysteresis to prevent jitter.

        Requires multiple consecutive fast frames before calling it "running".
        Also, once running, requires dropping below threshold to stop.

        Args:
            track_id: Track ID
            velocity: Smoothed velocity

        Returns:
            (is_running, confidence)
        """
        is_fast = velocity > self.running_threshold

        if is_fast:
            self._fast_frames[track_id] += 1
            if self._fast_frames[track_id] >= self.running_min_frames:
                self._was_running[track_id] = True
                confidence = min(1.0, velocity / (self.running_threshold * 1.5))
                return True, confidence
        else:
            # Reset counter if slow
            self._fast_frames[track_id] = 0
            # If was running, keep running until drops well below threshold
            if self._was_running[track_id]:
                # Hysteresis: stop running only when 20% below threshold
                if velocity < self.running_threshold * 0.8:
                    self._was_running[track_id] = False
                    return False, 0.0
                else:
                    # Still close to threshold, stay running
                    return True, 0.5

        return False, 0.0

    def _classify(
        self,
        track_id: int,
        velocity: float,
        aspect_ratio: float,
        vertical_velocity: float
    ) -> Tuple[str, float]:
        """
        Classify behavior based on rules.

        Returns: (behavior, confidence)
        """
        # Step 1: Check for falling (highest priority)
        is_falling, fall_confidence = self._detect_fall(
            track_id, aspect_ratio, vertical_velocity
        )
        if is_falling:
            return "falling", fall_confidence

        # Step 2: Check if already fallen (lying down but not actively falling)
        if aspect_ratio < self.fall_ar_threshold:
            return "fallen", 0.8

        # Step 3: Check for running (with hysteresis)
        is_running, run_confidence = self._classify_running(track_id, velocity)
        if is_running:
            return "running", run_confidence

        # Step 4: Check for walking
        if self.walking_min <= velocity <= self.walking_max:
            # Confidence peaks in middle of range
            mid = (self.walking_min + self.walking_max) / 2
            confidence = 1.0 - abs(velocity - mid) / (self.walking_max - self.walking_min)
            confidence = max(0.5, min(1.0, confidence))
            return "walking", confidence

        # Step 5: Standing still
        if velocity < self.walking_min * 0.5:
            return "standing", 0.9

        # In between walking and running
        return "walking", 0.5

    def analyze(self, tracks) -> List[BehaviorResult]:
        """
        Analyze tracks and classify behaviors.

        Args:
            tracks: List of Track objects from TrackingModule

        Returns:
            List of BehaviorResult
        """
        results = []
        current_time = time.time()

        for track in tracks:
            # Calculate aspect ratio
            x1, y1, x2, y2 = track.bbox
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            aspect_ratio = height / width

            # Get center position
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Calculate velocities
            instant_velocity = self._calculate_instant_velocity(
                track.track_id, (cx, cy)
            )
            smoothed_velocity = self._get_smoothed_velocity(
                track.track_id, instant_velocity
            )
            vertical_velocity = self._calculate_vertical_velocity(
                track.track_id, (cx, cy)
            )

            # Store data for next frame
            self._track_data[track.track_id].append(
                (current_time, (cx, cy), aspect_ratio, smoothed_velocity)
            )

            # Classify behavior
            behavior, confidence = self._classify(
                track.track_id,
                smoothed_velocity,
                aspect_ratio,
                vertical_velocity
            )

            result = BehaviorResult(
                track_id=track.track_id,
                behavior=behavior,
                confidence=confidence,
                velocity=smoothed_velocity,
                aspect_ratio=aspect_ratio
            )
            results.append(result)

        # Cleanup old tracks
        active_ids = {t.track_id for t in tracks}
        for tid in list(self._track_data.keys()):
            if tid not in active_ids:
                del self._track_data[tid]
                del self._velocity_history[tid]
                del self._fast_frames[tid]
                del self._was_running[tid]

        return results

    def reset(self):
        """Reset all stored track data."""
        self._track_data.clear()
        self._velocity_history.clear()
        self._fast_frames.clear()
        self._was_running.clear()


if __name__ == "__main__":
    """Test behavior analyzer with improved detection."""
    from src.tracking_module.tracker import Track

    print("Testing Improved BehaviorAnalyzer...")

    analyzer = BehaviorAnalyzer()

    # Test 1: Walking (smooth velocity)
    print("\n=== Test 1: Walking ===")
    for i in range(10):
        track = Track(
            track_id=1,
            bbox=(100 + i * 3, 100, 150 + i * 3, 250),
            confidence=0.9,
            history=[(125 + j * 3, 175) for j in range(i + 1)],
            age=i
        )
        behaviors = analyzer.analyze([track])
        if i >= 5:  # Wait for smoothing
            b = behaviors[0]
            print(f"Frame {i}: {b.behavior} (v={b.velocity:.1f}px/s, ar={b.aspect_ratio:.1f})")

    analyzer.reset()

    # Test 2: Running (requires consecutive fast frames)
    print("\n=== Test 2: Running (needs 3+ fast frames) ===")
    for i in range(6):
        track = Track(
            track_id=2,
            bbox=(100 + i * 8, 100, 150 + i * 8, 250),
            confidence=0.9,
            history=[(125 + j * 8, 175) for j in range(i + 1)],
            age=i
        )
        behaviors = analyzer.analyze([track])
        b = behaviors[0]
        print(f"Frame {i}: {b.behavior} (v={b.velocity:.1f}px/s)")

    analyzer.reset()

    # Test 3: Falling (AR drop + vertical velocity)
    print("\n=== Test 3: Falling (AR drop + downward movement) ===")
    for i in range(6):
        if i < 3:
            # Standing
            bbox = (200, 100, 250, 300)
        elif i == 3:
            # Starting to fall (AR drops)
            bbox = (200, 200, 300, 250)
        else:
            # Fallen
            bbox = (200, 250, 350, 300)

        track = Track(
            track_id=3,
            bbox=bbox,
            confidence=0.9,
            history=[(225 + i * 2, 200 + i * 5) for _ in range(i + 1)],
            age=i
        )
        behaviors = analyzer.analyze([track])
        b = behaviors[0]
        print(f"Frame {i}: {b.behavior} (ar={b.aspect_ratio:.1f}, v={b.velocity:.1f}px/s)")

    # Test 4: Not falling (AR stable)
    print("\n=== Test 4: Walking (not falling - AR stable) ===")
    analyzer.reset()
    for i in range(5):
        # Walking normally, AR stays around 2-3
        track = Track(
            track_id=4,
            bbox=(100 + i * 5, 100 + i * 2, 150 + i * 5, 250 + i * 2),
            confidence=0.9,
            history=[(125 + j * 5, 175 + j * 2) for j in range(i + 1)],
            age=i
        )
        behaviors = analyzer.analyze([track])
        b = behaviors[0]
        print(f"Frame {i}: {b.behavior} (ar={b.aspect_ratio:.1f}, v={b.velocity:.1f}px/s)")

    print("\nBehavior analyzer test complete!")
    print("\nImprovements:")
    print("- Smoothed velocity (5-frame average)")
    print("- Fall detection requires AR drop + vertical movement")
    print("- Running needs 3+ consecutive fast frames (hysteresis)")
