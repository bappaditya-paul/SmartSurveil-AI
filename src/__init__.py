"""Real-Time Intelligent Surveillance System - Week 1"""

__version__ = "0.1.0"
__phase__ = "Week 1: Input + Detection"

# Week 1 exports only
from .input_module import InputModule, Frame
from .detection_module import DetectionModule, DetectionResult

__all__ = [
    "InputModule",
    "Frame",
    "DetectionModule",
    "DetectionResult",
]
