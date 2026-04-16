# Module Specifications

## Overview

This document provides detailed specifications for each module in the Real-Time Intelligent Surveillance System. Each module is designed as an independent unit with clear interfaces, enabling testing, replacement, and extension.

---

## Module 1: Input Module

### Purpose
Acquire video from various sources and prepare frames for downstream processing.

### Responsibilities
- Video stream connection and management
- Frame capture and decoding
- Preprocessing (resize, color conversion)
- Frame rate control
- Resource cleanup

### Class Interface

```python
class InputModule:
    def __init__(self, source: Union[int, str], config: dict)
    def start(self) -> None
    def read(self) -> Optional[Frame]
    def get_fps(self) -> float
    def release(self) -> None
    def is_active(self) -> bool
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | int/str | 0 | Camera index or video path/RTSP URL |
| `target_resolution` | tuple | (640, 480) | Output frame dimensions |
| `buffer_size` | int | 1 | OpenCV internal buffer (1 = minimal latency) |
| `fps_limit` | float | None | Optional FPS cap |

### Input/Output Specification

**Input:** None (self-contained, connects to external source)

**Output:**
```python
@dataclass
class Frame:
    data: np.ndarray          # (H, W, 3) uint8
    timestamp: float          # time.time()
    source_id: str
    sequence_number: int      # Frame counter
```

### Error Handling

| Error | Response |
|-------|----------|
| Camera disconnected | Retry 3x → Raise ConnectionError |
| Decode failure | Skip frame, log warning |
| Resolution mismatch | Resize to target, log info |

### Threading Model

- Primary capture runs in dedicated thread
- `read()` method is blocking but thread-safe
- Double-buffering to prevent frame tearing

---

## Module 2: Detection Module

### Purpose
Detect humans in video frames using YOLOv8.

### Responsibilities
- Neural network inference
- Detection post-processing (NMS)
- Class filtering
- Coordinate normalization

### Class Interface

```python
class DetectionModule:
    def __init__(self, model_path: str, config: dict)
    def load_model(self) -> None
    def detect(self, frame: np.ndarray) -> List[Detection]
    def get_inference_time(self) -> float
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | "yolov8n.pt" | Path to YOLOv8 weights |
| `confidence_threshold` | float | 0.5 | Minimum detection confidence |
| `iou_threshold` | float | 0.45 | NMS IoU threshold |
| `classes` | List[int] | [0] | COCO class indices (0=person) |
| `device` | str | "auto" | "cpu", "cuda", or "auto" |
| `half_precision` | bool | False | FP16 inference (faster on GPU) |

### Input/Output Specification

**Input:** `np.ndarray` - BGR image from InputModule

**Output:**
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                # 0.0 - 1.0
    class_id: int
    class_name: str

# Module returns: List[Detection]
```

### Processing Pipeline

1. **Preprocessing**: Resize to model input size (640×640), normalize
2. **Inference**: Forward pass through YOLOv8
3. **Postprocessing**:
   - Decode predictions
   - Filter by confidence threshold
   - Apply NMS to remove duplicates
   - Filter by class (person only)
   - Scale bboxes back to original frame dimensions

### Performance Characteristics

| Model | Size | Speed (CPU) | Speed (GPU) | mAP |
|-------|------|-------------|-------------|-----|
| YOLOv8n | 6M params | ~20 FPS | ~100 FPS | 37.3% |
| YOLOv8s | 11M params | ~10 FPS | ~80 FPS | 44.9% |
| YOLOv8m | 26M params | ~5 FPS | ~50 FPS | 50.2% |

*Speed measured at 640×480 input on reference hardware*

---

## Module 3: Tracking Module

### Purpose
Maintain consistent identity for detected persons across frames using DeepSORT.

### Responsibilities
- Kalman filter-based motion prediction
- Appearance feature extraction
- Detection-to-track association
- Track lifecycle management
- ID assignment and recycling

### Class Interface

```python
class TrackingModule:
    def __init__(self, config: dict)
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]
    def get_active_tracks(self) -> List[Track]
    def get_track_history(self, track_id: int) -> Optional[deque]
    def reset(self) -> None
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | int | 30 | Max frames to keep lost tracks |
| `min_hits` | int | 3 | Min hits to confirm track |
| `iou_threshold` | float | 0.3 | Matching threshold |
| `feature_budget` | int | 100 | Appearance feature samples per track |
| `nn_matching_threshold` | float | 0.5 | Cosine distance threshold |

### Input/Output Specification

**Input:**
- `detections`: List[Detection] from DetectionModule
- `frame`: np.ndarray (original image for feature extraction)

**Output:**
```python
@dataclass
class Track:
    track_id: int                    # Persistent ID (1, 2, 3, ...)
    bbox: Tuple[int, int, int, int]   # Current position
    confidence: float
    state: TrackState                  # TENTATIVE | CONFIRMED | DELETED
    age: int                         # Total frames since creation
    time_since_update: int           # Frames since last detection
    history: deque[Tuple[int, int]]  # Centroid positions (last N)
    velocity: Tuple[float, float]     # Estimated velocity (px/frame)
    feature_vector: np.ndarray       # Appearance embedding

class TrackState(Enum):
    TENTATIVE = 1    # New track, not yet confirmed
    CONFIRMED = 2    # Active, reliable track
    DELETED = 3      # Lost for too long, removed
```

### Track Lifecycle

```
Detection ──▶ TENTATIVE ──[3 hits]──▶ CONFIRMED ──[lost]──▶ LOST
                                             │                 │
                                             │                 [max_age exceeded]
                                             │                 ▼
                                             │               DELETED
                                             │
                                             └──[update]──▶ CONFIRMED
```

### Algorithm Overview

1. **Prediction**: Kalman filter predicts next position for all tracks
2. **Matching**: Three-stage cascade:
   - Stage 1: Match by appearance + motion (confirmed tracks)
   - Stage 2: Match by IoU (confirmed tracks, unmatched detections)
   - Stage 3: Match by IoU (tentative tracks)
3. **Update**: Update matched tracks, create new tracks for unmatched detections
4. **Cleanup**: Mark old tracks as deleted

---

## Module 4: Behavior Analysis Module

### Purpose
Analyze movement patterns to classify behaviors and detect anomalies.

### Responsibilities
- Motion feature extraction
- Behavior classification (rule-based or ML)
- Alert condition evaluation
- Feature vector management for ML

### Class Interface

```python
class BehaviorAnalysisModule:
    def __init__(self, method: str, config: dict)
    def analyze(self, track: Track) -> BehaviorResult
    def analyze_batch(self, tracks: List[Track]) -> List[BehaviorResult]
    def set_alert_rules(self, rules: dict) -> None
    def train_ml_model(self, training_data: list) -> None  # If ML mode
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "rule_based" | "rule_based" or "ml_based" |
| `history_window` | int | 30 | Frames to analyze (1 sec @ 30fps) |
| `velocity_walking_threshold` | float | 5.0 | px/frame threshold |
| `velocity_running_threshold` | float | 10.0 | px/frame threshold |
| `fall_aspect_ratio_threshold` | float | 1.0 | height/width threshold |
| `loitering_time_threshold` | int | 300 | Frames before loitering (10 sec) |
| `alert_cooldown` | int | 150 | Frames between same-type alerts (5 sec) |

### Input/Output Specification

**Input:** `Track` object with position history

**Output:**
```python
@dataclass
class BehaviorResult:
    track_id: int
    behavior: BehaviorType          # Enum of behavior classes
    confidence: float               # 0.0 - 1.0
    features: MotionFeatures
    alert_triggered: bool
    alert_type: Optional[str]
    timestamp: float

class BehaviorType(Enum):
    UNKNOWN = 0
    WALKING = 1
    RUNNING = 2
    FALLING = 3
    LOITERING = 4
    SUSPICIOUS = 5

@dataclass
class MotionFeatures:
    velocity_mean: float
    velocity_std: float
    velocity_max: float
    acceleration_mean: float
    aspect_ratio_current: float
    aspect_ratio_change: float
    direction_variance: float
    position_variance: float
    trajectory_length: float
```

### Rule-Based Classification Logic

```python
def classify_behavior(features: MotionFeatures) -> BehaviorType:
    # Falling detection (highest priority)
    if features.aspect_ratio_change < -0.5 and \
       features.aspect_ratio_current < 1.0:
        return BehaviorType.FALLING

    # Running detection
    if features.velocity_mean > velocity_running_threshold or \
       features.velocity_max > velocity_running_threshold * 1.5:
        return BehaviorType.RUNNING

    # Walking detection
    if features.velocity_mean > velocity_walking_threshold:
        return BehaviorType.WALKING

    # Loitering detection
    if features.position_variance < 100 and \
       track_age > loitering_time_threshold:
        return BehaviorType.LOITERING

    # Suspicious activity
    if features.direction_variance > threshold:
        return BehaviorType.SUSPICIOUS

    return BehaviorType.UNKNOWN
```

### ML-Based Approach (Optional)

When `method="ml_based"`:
1. Extract 25-dimension feature vector from track history
2. Pass through trained classifier (Random Forest / Neural Network)
3. Return predicted class with probability

---

## Module 5: Output Module

### Purpose
Present system results through multiple channels: visualization, logging, and alerting.

### Responsibilities
- Frame annotation and rendering
- Dashboard updates
- Event logging
- Alert generation
- Heatmap accumulation and display

### Class Interface

```python
class OutputModule:
    def __init__(self, config: dict)
    def render(self, frame: np.ndarray, tracks: List[Track], 
               behaviors: List[BehaviorResult]) -> np.ndarray
    def update_dashboard(self, metrics: dict) -> None
    def log_event(self, event: dict) -> None
    def trigger_alert(self, alert: dict) -> None
    def update_heatmap(self, positions: List[Tuple[int, int]]) -> None
    def get_heatmap(self) -> np.ndarray
    def save_snapshot(self, frame: np.ndarray, filename: str) -> None
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `display_enabled` | bool | True | Show OpenCV window |
| `dashboard_enabled` | bool | True | Launch Streamlit |
| `log_enabled` | bool | True | Write JSON logs |
| `log_directory` | str | "logs/" | Log file location |
| `alert_webhook` | str | None | URL for alert callbacks |
| `heatmap_enabled` | bool | True | Generate activity heatmap |
| `heatmap_resolution` | tuple | (64, 48) | Grid resolution |
| `save_alert_clips` | bool | False | Save video on alert |

### Input/Output Specification

**Input:**
- Original frame (np.ndarray)
- List of Track objects
- List of BehaviorResult objects
- System metrics dict

**Output Channels:**

| Channel | Format | Destination |
|---------|--------|-------------|
| Display | np.ndarray (BGR) | cv2.imshow() or Streamlit |
| Log | JSON Lines | logs/events_YYYY-MM-DD.jsonl |
| Alert | JSON payload | Dashboard + optional webhook |
| Heatmap | np.ndarray (float32) | Dashboard overlay |

### Visual Annotation Schema

```python
# Color coding
COLORS = {
    "bbox_default": (0, 255, 0),      # Green
    "bbox_running": (0, 0, 255),      # Red
    "bbox_falling": (0, 0, 255),      # Red + flashing
    "bbox_loitering": (0, 165, 255),  # Orange
    "text": (255, 255, 255),          # White
    "trajectory": (255, 0, 0),        # Blue line
}

# Text format: "ID:5 | RUNNING | 89%"
```

### Log Entry Format

```json
{
    "timestamp": "2024-04-13T10:30:00.123Z",
    "event_type": "behavior_detection",
    "track_id": 5,
    "behavior": "falling",
    "confidence": 0.89,
    "position": [320, 240],
    "frame_number": 1234,
    "alert_triggered": true
}
```

### Alert Escalation

```
Behavior Detected ──▶ Check Alert Rules ──▶ Check Cooldown
                                         │
                                         Match?
                                          │
                                          Yes
                                          ▼
                              ┌──────────────────┐
                              │ Generate Alert   │
                              │ • Dashboard      │
                              │ • Log entry      │
                              │ • Webhook (opt)  │
                              └──────────────────┘
```

---

## Module Interactions Summary

```
┌────────────┐    Frame    ┌────────────┐   Detections   ┌────────────┐
│   Input    │ ───────────▶│ Detection  │ ──────────────▶│  Tracking  │
│   Module   │             │   Module   │                │   Module   │
└────────────┘             └────────────┘                └─────┬──────┘
                                                               │
                                                               │ Tracks
                                                               ▼
                                                        ┌────────────┐
                                                        │  Behavior  │
                                                        │  Analysis  │
                                                        └─────┬──────┘
                                                              │
                                                              │ Results
                                                              ▼
                                                       ┌────────────┐
                                                       │   Output   │
                                                       │   Module   │
                                                       └────────────┘
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Current | Initial module specifications |
