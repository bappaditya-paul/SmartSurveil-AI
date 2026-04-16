# Complete Run Guide - AI Surveillance System

## Quick Start (3 Steps)

### Step 1: Activate Virtual Environment

**Windows:**
```cmd
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 2: Run Test (Verify Installation)

```bash
python test_system.py
```

You should see:
```
============================================================
AI SURVEILLANCE SYSTEM - FULL TEST
============================================================
[1/5] Creating OutputModule...
[2/5] Creating test data...
[3/5] Processing frames...
[4/5] Checking metrics...
[5/5] Checking alerts...
============================================================
ALL TESTS PASSED!
============================================================
```

### Step 3: Run Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Dashboard Usage

### Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  AI Surveillance Dashboard         [Settings Sidebar]       │
│                                                             │
│  ┌──────────────────────────┐    ┌──────────┬─────────┐    │
│  │                          │    │   FPS    │  25.5   │    │
│  │    📹 LIVE VIDEO FEED   │    ├──────────┼─────────┤    │
│  │                          │    │  Tracks  │    3    │    │
│  │  [Bounding boxes with    │    ├──────────┼─────────┤    │
│  │   IDs and behaviors]     │    │  Alerts  │    1    │    │
│  │                          │    └──────────┴─────────┘    │
│  │  ID:1 walking  🟢        │                              │
│  │  ID:2 falling  🔴        │    🚨 RECENT ALERTS          │
│  │                          │    ┌────────────────────┐    │
│  │  🔴 Alert: FALLING!      │    │ 🔴 FALLING_DETECTED│    │
│  │                          │    │ 🟡 RUNNING_DETECTED│    │
│  └──────────────────────────┘    └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Controls

1. **Settings Sidebar (Left)**
   - **Video Source**: Choose "Webcam" or "Video File"
   - **Confidence Slider**: Adjust detection threshold (0.1 - 1.0)
   - **Start Button**: Begin surveillance
   - **Stop Button**: Stop surveillance

2. **Live Metrics (Right)**
   - **FPS**: System performance (aim for >20)
   - **Active Tracks**: People currently being tracked
   - **Alerts**: Current active alerts

3. **Alerts Table**
   - Shows recent alerts with severity
   - 🔴 = High (falling), 🟡 = Medium, 🟢 = Low

---

## Running with Video File

### Method 1: Edit Dashboard

Open `app/dashboard.py` and change:
```python
source = st.sidebar.text_input(
    "Video Path",
    value="data/sample_video.mp4",  # <-- Change this path
    help="Path to video file"
)
```

### Method 2: Direct Run (No Dashboard)

Create a script `run_video.py`:

```python
import sys
sys.path.insert(0, '.')

import cv2
from src.input_module.input_handler import InputModule
from src.detection_module.detector import DetectionModule
from src.tracking_module.tracker import TrackingModule
from src.behavior_module.behavior_analyzer import BehaviorAnalyzer
from src.output_module.output_handler import OutputModule

# Initialize modules
input_module = InputModule(source="path/to/your/video.mp4")
detector = DetectionModule(classes=[0])  # Person only
tracker = TrackingModule()
behavior = BehaviorAnalyzer()
output = OutputModule()

input_module.open()
detector.warmup()

while True:
    frame_obj = input_module.read()
    if frame_obj is None:
        break
    
    frame = frame_obj.image
    
    # Pipeline
    detections = detector.detect(frame)
    person_detections = [d for d in detections if d.class_name == "person"]
    tracks = tracker.update(person_detections, frame)
    behaviors = behavior.analyze(tracks)
    output_frame = output.process(frame, tracks, behaviors)
    
    # Show
    cv2.imshow('Surveillance', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

input_module.close()
cv2.destroyAllWindows()
```

Run it:
```bash
python run_video.py
```

---

## Troubleshooting

### Issue: "streamlit: command not found"

**Fix:** Install streamlit
```bash
pip install streamlit pandas matplotlib
```

### Issue: "No module named 'src'"

**Fix:** Run from project root directory
```bash
cd E:\Coding\AI-Intelligent-Surveillance-System
streamlit run app/dashboard.py
```

### Issue: Webcam not working

**Fix:** Check camera index
```python
# Try different indices: 0, 1, 2, etc.
input_module = InputModule(source=1)  # Try 1 instead of 0
```

### Issue: "Module not found: deep_sort_realtime"

**Fix:** Install tracking dependency
```bash
pip install deep-sort-realtime
```

### Issue: "Module not found: ultralytics"

**Fix:** Install detection dependency
```bash
pip install ultralytics
```

---

## Understanding the Output

### Colors in Video

| Color  | Meaning                      |
|--------|------------------------------|
| 🟢 Green | Walking                    |
| 🟠 Orange | Running                   |
| 🔴 Red | Falling/Fallen (ALERT!)    |
| 🔵 Cyan | Standing                   |

### JSON Logs

Logs are saved to `logs/events_YYYYMMDD_HHMMSS.jsonl`:

```json
{"event_type": "behavior", "track_id": 1, "behavior": "walking", ...}
{"event_type": "alert", "alert_type": "FALLING_DETECTED", "severity": "high", ...}
```

Each line is one JSON event. View with:
```bash
cat logs/events_*.jsonl | head -10
```

---

## Project Structure

```
AI-Intelligent-Surveillance-System/
│
├── app/
│   └── dashboard.py          # Streamlit dashboard
│
├── src/
│   ├── input_module/
│   │   └── input_handler.py  # Video input
│   ├── detection_module/
│   │   └── detector.py        # YOLO detection
│   ├── tracking_module/
│   │   └── tracker.py          # DeepSORT tracking
│   ├── behavior_module/
│   │   └── behavior_analyzer.py # Behavior rules
│   └── output_module/
│       ├── output_handler.py   # Main output module
│       ├── visualizer.py       # Drawing functions
│       └── logger.py           # JSON logging
│
├── logs/                      # Event logs (created at runtime)
├── test_system.py             # System test
├── test_output.jpg            # Test visualization
└── RUN_GUIDE.md               # This file
```

---

## Next Steps

After Phase 1 is working:
1. Collect training data from logs
2. Train ML behavior classifier (Phase 2)
3. Add heatmap visualization
4. Add historical analysis

---

## Support

If something doesn't work:
1. Run `python test_system.py` - checks all components
2. Check log files in `logs/` directory
3. Verify all dependencies: `pip list | grep -E "streamlit|ultralytics|deep"`

**Happy surveillance!** 🎥
