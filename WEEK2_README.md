# Week 2: Tracking + Behavior Pipeline

This extends Week 1 with multi-person tracking and behavior classification.

## What's New

### 1. TrackingModule (`src/tracking_module/`)
- **DeepSORT** tracking for persistent person IDs
- Maintains 30-frame position history per person
- Handles occlusions and re-identification

### 2. BehaviorModule (`src/behavior_module/`)
- **Rule-based** behavior classification:
  - **Walking**: velocity 2-8 px/frame
  - **Running**: velocity > 8 px/frame
  - **Falling**: aspect ratio drops suddenly
  - **Standing**: low velocity

### 3. Updated Pipeline
```
Frame → Detection → Tracking → Behavior → Display
        [YOLOv8]    [DeepSORT]   [Rules]    [Vis]
```

## File Structure Added

```
src/
├── tracking_module/
│   ├── __init__.py
│   └── tracker.py          # DeepSORT wrapper + Track class
├── behavior_module/
│   ├── __init__.py
│   ├── behavior_analyzer.py   # Rule-based classifier
│   └── rule_engine.py        # Threshold definitions
└── utils/
    └── visualization.py     # Updated: draw tracks + behaviors

app/
└── main.py                 # Updated: full pipeline
```

## Installation

```bash
# Already installed from Week 1, just add DeepSORT:
pip install deep-sort-realtime
```

## How to Run

```bash
# Webcam
python app/main.py

# Video file
python app/main.py --source path/to/video.mp4
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `r` | Reset tracker (clear all IDs) |

## Display Info

- **Green box**: Walking person
- **Red box**: Running/Falling person
- **Cyan box**: Standing person
- **Label**: `ID:{id} {behavior}(confidence)`
- **Trajectory**: Blue line showing past positions

## Behavior Rules Explained

```python
# In behavior_analyzer.py

WALKING:  2.0 <= velocity <= 8.0 px/frame
RUNNING:  velocity > 8.0 px/frame
FALLING:  aspect_ratio < 1.0 (was > 2.0)
STANDING: velocity < 2.0 px/frame
```

**Velocity**: Calculated from center point movement between frames.

**Aspect Ratio**: `height / width` of bounding box.
- Standing: AR ~ 2-3 (tall and narrow)
- Fallen: AR ~ 0.5-1 (short and wide)

## Code Example

```python
from src.input_module import InputModule
from src.detection_module import DetectionModule
from src.tracking_module import TrackingModule
from src.behavior_module import BehaviorAnalyzer

# Initialize
input_module = InputModule(source=0)
detector = DetectionModule(classes=[0])  # Person only
tracker = TrackingModule(max_age=30)
analyzer = BehaviorAnalyzer()

# Pipeline
for frame in input_module.stream():
    detections = detector.detect(frame.image)
    tracks = tracker.update(detections, frame.image)
    behaviors = analyzer.analyze(tracks)

    for behavior in behaviors:
        print(f"Person {behavior.track_id}: {behavior.behavior}")
        print(f"  Velocity: {behavior.velocity:.1f} px/frame")
        print(f"  Aspect Ratio: {behavior.aspect_ratio:.2f}")
```

## Key Classes

### Track (from TrackingModule)
```python
@dataclass
class Track:
    track_id: int          # Persistent ID (1, 2, 3...)
    bbox: tuple            # (x1, y1, x2, y2)
    confidence: float
    history: List[(x, y)]  # Past center positions
    age: int               # Frames since birth
```

### BehaviorResult (from BehaviorAnalyzer)
```python
@dataclass
class BehaviorResult:
    track_id: int
    behavior: str          # "walking", "running", "falling", "standing"
    confidence: float      # 0.0 - 1.0
    velocity: float        # px/frame
    aspect_ratio: float    # height/width
```

## Testing Individual Components

```bash
# Test tracker
python -m src.tracking_module.tracker

# Test behavior analyzer
python -m src.behavior_module.behavior_analyzer
```

## Troubleshooting

### "No module named deep_sort_realtime"
```bash
pip install deep-sort-realtime
```

### Tracks jump around
- Increase `max_age` in tracker config
- Check lighting (poor light = poor ReID)
- Reduce `iou_threshold` slightly

### Behavior classification wrong
- Adjust velocity thresholds for your camera FPS
- Check aspect ratio values printed in console
- Modify rules in `behavior_analyzer.py`

### First run is slow
- DeepSORT downloads MobileNet ReID model on first run
- ~20MB download, happens once

## Performance

Typical FPS on CPU:
- Week 1 (Detection only): ~15-25 FPS
- Week 2 (+ Tracking): ~10-20 FPS
- Week 2 (+ Behavior): ~10-18 FPS

To improve:
- Lower resolution (config.yaml)
- Use GPU (set device: "cuda")
- Reduce history_size

## Next Week (Week 3)

- Event logging (JSON)
- Heatmap generation
- Streamlit dashboard
