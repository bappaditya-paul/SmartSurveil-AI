# Week 1: Input + Detection Pipeline

This is the first phase of the Intelligent Surveillance System. It captures video input and detects persons using YOLOv8.

## What This Does

1. **InputModule**: Captures video from webcam or file
2. **DetectionModule**: Detects persons using YOLOv8
3. **Visualization**: Draws bounding boxes around detected persons

## File Structure

```
.
├── app/
│   └── main.py                 # Entry point - runs the pipeline
├── config/
│   └── config.yaml             # Settings (source, thresholds, etc.)
├── src/
│   ├── __init__.py
│   ├── input_module/
│   │   ├── __init__.py
│   │   └── input_handler.py    # Video capture logic
│   ├── detection_module/
│   │   ├── __init__.py
│   │   └── detector.py         # YOLOv8 wrapper
│   └── utils/
│       ├── config_loader.py    # YAML config reader
│       └── visualization.py    # Drawing functions
├── requirements.txt            # Python dependencies
└── WEEK1_README.md            # This file
```

## Installation

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: First run will download YOLOv8n (~6MB) automatically.

## How to Run

### Option 1: Webcam (Default)
```bash
python app/main.py
```

### Option 2: Video File
```bash
python app/main.py --source "path/to/your/video.mp4"
```

### Option 3: Custom Config
```bash
python app/main.py --config config/config.yaml --source 0
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |

## Configuration

Edit `config/config.yaml`:

```yaml
input:
  source: 0                    # 0=webcam, or file path
  width: 640                   # Frame width
  height: 480                  # Frame height

detection:
  model_path: "yolov8n.pt"     # Model file
  confidence_threshold: 0.5    # Detection threshold (0-1)
  classes: [0]                 # 0 = person only

visualization:
  bbox_color: [0, 255, 0]      # Green boxes (BGR)
  show_fps: true               # Show FPS counter
```

## Testing Individual Modules

### Test Input Module Only
```bash
python -m src.input_module.input_handler
```

### Test Detection Module Only
```bash
python -m src.detection_module.detector
```

## Expected Output

When running, you should see:
- A window showing video feed
- Green bounding boxes around detected persons
- FPS counter in top-left
- Frame info at bottom

Example console output:
```
==================================================
Week 1: Input + Detection Pipeline
==================================================
Source: Webcam (0)

[1/3] Initializing InputModule...
Input opened: 640x480 @ 30.0fps

[2/3] Initializing DetectionModule...
Loading YOLOv8 model: yolov8n.pt
Model loaded on device: cpu
Model warmed up

[3/3] Starting pipeline...
Press 'q' to quit, 'p' to pause

Frame 30: 2 persons, FPS: 15.3, Inference: 45.2ms
Frame 60: 1 persons, FPS: 15.1, Inference: 43.8ms
```

## Troubleshooting

### "Could not open video source"
- Check webcam is connected
- Try different source index: `--source 1` instead of `0`
- Verify video file path is correct

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "CUDA out of memory"
- Edit config: set `device: "cpu"`
- Or reduce resolution in config

### Model downloads slowly
- First run downloads YOLOv8n (~6MB)
- Check internet connection
- Can manually download and place in project root

## Next Steps (Week 2)

- Add TrackingModule (DeepSORT) for person IDs
- Add BehaviorModule for action classification
- Connect OutputModule for logging

## Code Explanation

### InputModule (`src/input_module/input_handler.py`)

```python
# Create capture from webcam
input_module = InputModule(source=0, width=640, height=480)

# Iterate over frames
for frame in input_module.stream():
    print(frame.image.shape)   # (480, 640, 3)
    print(frame.timestamp)       # 1234567890.123
    print(frame.frame_number)    # 1, 2, 3, ...
```

### DetectionModule (`src/detection_module/detector.py`)

```python
# Initialize detector
detector = DetectionModule(
    model_path="yolov8n.pt",
    confidence_threshold=0.5,
    classes=[0]  # Person only
)

# Detect on a frame
detections = detector.detect(frame.image)
for det in detections:
    print(det.bbox)          # (x1, y1, x2, y2)
    print(det.confidence)    # 0.85
    print(det.class_name)    # "person"
```

## Performance Tips

1. **Lower resolution** = faster processing
2. **YOLOv8n** (nano) is fastest, try YOLOv8s for better accuracy
3. **GPU** (CUDA) is much faster than CPU
4. **Confidence threshold** tuning: higher = fewer false positives

## Research Notes

- YOLOv8n runs at ~15-30 FPS on CPU (640x480)
- Detection accuracy depends on lighting and distance
- Bounding boxes are in pixel coordinates
- Timestamp is Unix epoch (seconds since 1970)
