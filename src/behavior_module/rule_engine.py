"""
Rule Engine

Defines classification rules for behavior analysis.
Simple container for rule thresholds.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BehaviorRules:
    """
    Configuration for behavior classification rules.

    All velocities in pixels per frame (at 30fps).
    Aspect ratios are height/width.
    """
    # Velocity thresholds
    walking_min_velocity: float = 2.0
    walking_max_velocity: float = 8.0
    running_min_velocity: float = 8.0

    # Falling detection
    falling_aspect_ratio_threshold: float = 1.0
    falling_aspect_ratio_drop: float = 1.0  # Must drop by this much
    falling_vertical_velocity: float = 5.0

    # History window
    velocity_smoothing_window: int = 3  # Frames to average


class RuleEngine:
    """
    Simple rule container with helper methods.

    Usage:
        rules = RuleEngine()

        # Check if velocity indicates walking
        if rules.is_walking(velocity):
            behavior = "walking"
    """

    def __init__(self, rules: Optional[BehaviorRules] = None):
        self.rules = rules or BehaviorRules()

    def is_walking(self, velocity: float) -> bool:
        """Check if velocity is in walking range."""
        return self.rules.walking_min_velocity <= velocity <= self.rules.walking_max_velocity

    def is_running(self, velocity: float) -> bool:
        """Check if velocity indicates running."""
        return velocity >= self.rules.running_min_velocity

    def is_fallen(self, aspect_ratio: float) -> bool:
        """Check if aspect ratio indicates fallen person."""
        return aspect_ratio < self.rules.falling_aspect_ratio_threshold

    def is_falling(self, aspect_ratio: float, prev_aspect_ratio: float, vertical_velocity: float) -> bool:
        """
        Check if person is in process of falling.

        Requires sudden drop in aspect ratio + downward motion.
        """
        ar_drop = prev_aspect_ratio - aspect_ratio
        if ar_drop < self.rules.falling_aspect_ratio_drop:
            return False

        return vertical_velocity > self.rules.falling_vertical_velocity

    def get_velocity_confidence(self, velocity: float, target: float, tolerance: float) -> float:
        """
        Calculate confidence based on how close velocity is to target.

        Returns 1.0 at target, decreasing linearly to 0 at tolerance.
        """
        diff = abs(velocity - target)
        return max(0.0, 1.0 - diff / tolerance)


# Default global rules instance
_default_rules = BehaviorRules()


def get_default_rules() -> BehaviorRules:
    """Get default rule configuration."""
    return _default_rules
