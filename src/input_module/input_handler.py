"""
Input Module

Handles video capture from webcam or video files.
Returns frames with timestamps for downstream processing.
"""

import cv2
import time
from dataclasses import dataclass
from typing import Union, Optional, Generator
import numpy as np


@dataclass
class Frame:
    """
    Container for a video frame with metadata.

    Attributes:
        image: The actual image data (numpy array)
        timestamp: Time when frame was captured (seconds since epoch)
        frame_number: Sequential frame index
        source: Source identifier (camera index or file path)
    """
    image: np.ndarray
    timestamp: float
    frame_number: int
    source: Union[int, str]


class InputModule:
    """
    Video capture handler for webcam or file input.

    Usage:
        # Webcam
        input_module = InputModule(source=0)

        # Video file
        input_module = InputModule(source="path/to/video.mp4")

        # Iterate over frames
        for frame in input_module.stream():
            process(frame.image)
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize video capture.

        Args:
            source: 0 for default webcam, or path to video file
            width: Target frame width (None = keep original)
            height: Target frame height (None = keep original)
            fps: Target FPS (None = keep original)
        """
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._is_opened = False

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if successfully opened, False otherwise
        """
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            print(f"ERROR: Could not open video source: {self.source}")
            return False

        # Apply target settings if specified
        if self.target_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        if self.target_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._is_opened = True
        self._frame_count = 0

        # Print actual settings being used
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"Input opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")

        return True

    def read(self) -> Optional[Frame]:
        """
        Read a single frame.

        Returns:
            Frame object or None if end of stream/error
        """
        if not self._is_opened or self._cap is None:
            return None

        ret, image = self._cap.read()

        if not ret:
            return None

        self._frame_count += 1

        return Frame(
            image=image,
            timestamp=time.time(),
            frame_number=self._frame_count,
            source=self.source
        )

    def stream(self) -> Generator[Frame, None, None]:
        """
        Generator that yields frames continuously.

        Usage:
            for frame in input_module.stream():
                process(frame)

        Yields:
            Frame objects until stream ends (file) or interrupted (webcam)
        """
        if not self._is_opened:
            if not self.open():
                return

        try:
            while True:
                frame = self.read()
                if frame is None:
                    # End of video file
                    break
                yield frame
        except KeyboardInterrupt:
            print("\nStream interrupted by user")
        finally:
            self.close()

    def get_fps(self) -> float:
        """Get the actual FPS of the video source."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def get_resolution(self) -> tuple:
        """Get the actual resolution (width, height)."""
        if self._cap is not None:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)

    def close(self):
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._is_opened = False
            print("Input module closed")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    """Test the input module - displays video with frame counter."""
    print("Testing InputModule...")
    print("Press 'q' to quit\n")

    # Try webcam (source=0)
    input_module = InputModule(source=0, width=640, height=480)

    try:
        for frame in input_module.stream():
            # Add frame info overlay
            cv2.putText(
                frame.image,
                f"Frame: {frame.frame_number}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Input Test", frame.image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
