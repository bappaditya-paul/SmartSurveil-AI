"""
Output Module for visualization, logging, and alerting.

This module handles all system outputs including real-time visualization,
JSON event logging, alert generation, and dashboard updates.

Example:
    from src.output_module import OutputModule

    output = OutputModule(log_dir="logs")

    for frame in video:
        output_frame = output.process(frame, tracks, behaviors)
        # Display output_frame

        # Get live metrics
        metrics = output.get_metrics()
        print(f"FPS: {metrics['fps']}, Tracks: {metrics['active_tracks']}")
"""

from .output_handler import OutputModule, Alert
from .visualizer import Visualizer
from .logger import EventLogger

__all__ = [
    "OutputModule",
    "Visualizer",
    "EventLogger",
    "Alert",
]
