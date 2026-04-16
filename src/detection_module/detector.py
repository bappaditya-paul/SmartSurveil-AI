"""
Detection Module

Person detection using YOLOv8 from ultralytics.
Returns detection results with bounding boxes and confidence scores.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import time


@dataclass
class DetectionResult:
    """
    Container for a single detection.

    Attributes:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        confidence: Detection confidence (0.0 - 1.0)
        class_id: COCO class ID (0 for person)
        class_name: Human-readable class name
    """
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class DetectionModule:
    """
    YOLOv8-based person detector.

    Usage:
        detector = DetectionModule(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )

        detections = detector.detect(frame)
        for det in detections:
            print(f"Person at {det.bbox}, confidence: {det.confidence:.2f}")
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
        device: str = "auto"
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path: Path to YOLOv8 weights file
                       (will auto-download if not found)
            confidence_threshold: Minimum confidence to keep detection
            classes: List of COCO class IDs to detect (None = all)
                     Use [0] for person-only detection
            device: "auto", "cpu", or "cuda" for GPU
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes  # None = detect all, [0] = person only
        self.device = self._resolve_device(device)

        self.model = None
        self._inference_time_ms = 0.0

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def initialize(self) -> bool:
        """
        Load the YOLOv8 model.

        Returns:
            True if successfully loaded
        """
        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Model loaded on device: {self.device}")
            return True

        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Make sure ultralytics is installed: pip install ultralytics")
            return False

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run detection on a single image.

        Args:
            image: OpenCV image (BGR format, numpy array)

        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            if not self.initialize():
                return []

        # Run inference
        start_time = time.time()
        results = self.model(
            image,
            verbose=False,  # Suppress console output
            conf=self.confidence_threshold,
            classes=self.classes
        )
        self._inference_time_ms = (time.time() - start_time) * 1000

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Get class name
                class_name = self.model.names[class_id]

                detection = DetectionResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

        return detections

    def get_last_inference_time(self) -> float:
        """Get the inference time of the last detection in milliseconds."""
        return self._inference_time_ms

    def warmup(self, image_size: tuple = (640, 480)):
        """
        Run a dummy inference to warm up the model.
        This helps get accurate timing for the first real detection.
        """
        if self.model is None:
            self.initialize()

        dummy_image = np.zeros((*image_size[::-1], 3), dtype=np.uint8)
        self.detect(dummy_image)
        print("Model warmed up")


if __name__ == "__main__":
    """Test the detection module on webcam feed."""
    import cv2
    from src.input_module.input_handler import InputModule

    print("Testing DetectionModule...")
    print("Press 'q' to quit\n")

    # Initialize modules
    input_module = InputModule(source=0, width=640, height=480)
    detector = DetectionModule(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        classes=[0]  # Person only
    )

    # Warmup
    detector.warmup()

    try:
        for frame in input_module.stream():
            # Detect persons
            detections = detector.detect(frame.image)

            # Draw results
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det.class_name}: {det.confidence:.2f}"
                cv2.putText(
                    frame.image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            # Show FPS and detection count
            cv2.putText(
                frame.image,
                f"Detections: {len(detections)} | Inference: {detector.get_last_inference_time():.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("Detection Test", frame.image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
