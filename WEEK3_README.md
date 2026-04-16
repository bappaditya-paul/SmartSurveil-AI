# Week 3: Output Module + Dashboard

## Overview

This week we add the **OutputModule** (visualization + logging) and a **Streamlit Dashboard** for real-time monitoring. This completes Phase 1 of our surveillance system.

## New Components

```
┌─────────────────────────────────────────────────────────────┐
│                    SURVEILLANCE PIPELINE (Phase 1)         │
├─────────────────────────────────────────────────────────────┤
│  Input → Detection → Tracking → Behavior → OUTPUT MODULE      │
│                                              ↓              │
│                                    ┌─────────────────────┐  │
│                                    │  Visualization      │  │
│                                    │  - Draw bbox        │  │
│                                    │  - Draw ID          │  │
│                                    │  - Draw behavior    │  │
│                                    └─────────────────────┘  │
│                                              ↓              │
│                                    ┌─────────────────────┐  │
│                                    │  JSON Logging       │  │
│                                    │  - Events stored    │  │
│                                    │  - Alert history    │  │
│                                    └─────────────────────┘  │
│                                              ↓              │
│                                    ┌─────────────────────┐  │
│                                    │  Streamlit UI       │  │
│                                    │  - Live video       │  │
│                                    │  - Metrics (FPS)    │  │
│                                    │  - Alerts table     │  │
│                                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 1. OutputModule

### Location
`src/output_module/output_handler.py`

### What It Does
- **Draws on frames**: Bounding boxes, track IDs, behavior labels
- **Logs events**: JSON format for detections, behaviors, and alerts
- **Generates alerts**: Triggers when suspicious behavior detected

### Code Example
```python
from src.output_module import OutputModule

output = OutputModule(log_dir="logs")

for frame in video:
    # Draw tracks, behaviors, alerts on frame
    output_frame = output.process(frame, tracks, behaviors)
    
    # Get live metrics
    metrics = output.get_metrics()
    print(f"FPS: {metrics['fps']:.1f}")
    print(f"Tracks: {metrics['active_tracks']}")
    print(f"Alerts: {metrics['active_alerts']}")
```

### Visualizer Features

| Behavior | Box Color | Example Label |
|----------|-----------|---------------|
| Walking  | 🟢 Green | `ID:1 walking` |
| Running  | 🟠 Orange | `ID:2 running` |
| Falling  | 🔴 Red | `ID:3 falling` |
| Fallen   | 🔴 Dark Red | `ID:3 fallen` |
| Standing | 🔵 Cyan | `ID:4 standing` |

### JSON Log Format

Each line in `logs/events_YYYYMMDD_HHMMSS.jsonl`:
```json
{"event_type": "behavior", "track_id": 1, "behavior": "walking", "confidence": 0.9, "velocity": 5.2, "timestamp": "2024-01-15T10:30:45.123456"}
{"event_type": "alert", "alert_type": "FALLING_DETECTED", "track_id": 2, "severity": "high", "timestamp": "2024-01-15T10:30:46.234567"}
```

## 2. Streamlit Dashboard

### Location
`app/dashboard.py`

### Run It
```bash
# Install dependencies (if not already installed)
pip install streamlit pandas matplotlib

# Run the dashboard
streamlit run app/dashboard.py
```

### Dashboard Interface

```
┌────────────────────────────────────────────────────────────┐
│  🎥 AI Surveillance Dashboard                              │
│  Real-time person tracking and behavior monitoring         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📹 Live Feed              │ 📊 Metrics                   │
│  ┌────────────────────┐    │ ┌──────────┬────────┬──────┐│
│  │                    │    │ │ FPS      │Tracks │Alerts││
│  │  [Live video]      │    │ │ 25.5     │   2   │  1   ││
│  │  Bounding boxes    │    │ └──────────┴────────┴──────┘│
│  │  with IDs          │    │                             │
│  │  and behaviors     │    │ 🚨 Recent Alerts            │
│  │                    │    │ ┌───────────────────────┐   │
│  │  🔴 Alert: Person 1│    │ │ 🔴 FALLING_DETECTED   │   │
│  │     Falling!       │    │ │    ID: 2              │   │
│  │                    │    │ │    10:30:46           │   │
│  └────────────────────┘    │ └───────────────────────┘   │
│                            │                             │
├────────────────────────────────────────────────────────────┤
│  ⚙️ Settings (Sidebar)                                     │
│  • Video Source: Webcam / Video File                        │
│  • Detection Confidence: 0.5                                │
│  • [Start] [Stop] buttons                                   │
└────────────────────────────────────────────────────────────┘
```

### Features

1. **Live Video**: Shows annotated video with bounding boxes, IDs, and behaviors
2. **Real-time Metrics**:
   - **FPS**: Frames per second (system performance)
   - **Active Tracks**: Number of people currently tracked
   - **Active Alerts**: Number of current alerts
3. **Alerts Table**: Recent alerts with:
   - Severity (🔴 High, 🟡 Medium, 🟢 Low)
   - Alert type (FALLING_DETECTED, etc.)
   - Track ID
   - Timestamp

## File Structure

```
src/output_module/
├── __init__.py           # Module exports
├── output_handler.py     # Main OutputModule class
├── visualizer.py         # Drawing functions (bbox, labels)
├── logger.py             # JSON event logging
└── heatmap_generator.py  # (stub for future)

app/
├── __init__.py
└── dashboard.py          # Streamlit dashboard

logs/                     # Created at runtime
└── events_*.jsonl        # Event log files
```

## How It Works (Student Explanation)

### Step-by-Step Flow

```
Frame from camera
      ↓
Detection (find people)
      ↓
Tracking (assign IDs: 1, 2, 3...)
      ↓
Behavior (walking? running? falling?)
      ↓
OUTPUT MODULE:
  1. Draw boxes + labels on frame
  2. Save events to JSON file
  3. Check if alert needed (e.g., "falling!")
      ↓
Show in Streamlit dashboard
```

### Alert System

Alerts trigger on specific behaviors:

```python
ALERT_BEHAVIORS = {
    "falling": ("FALLING_DETECTED", "high"),
    "fallen":  ("PERSON_FALLEN", "high"),
    "running": ("RUNNING_DETECTED", "low"),
}
```

**Cooldown**: Same alert type can't trigger again for 5 seconds (prevents spam).

## Running the Complete System

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Dashboard
```bash
streamlit run app/dashboard.py
```

### Step 3: Use Dashboard
1. Select video source (webcam or file)
2. Adjust confidence slider if needed
3. Click **Start**
4. Watch live feed with detections
5. Check alerts table for suspicious behavior

## Testing Components

### Test Visualizer
```bash
cd src/output_module
python visualizer.py
```
Creates `test_visualization.jpg` showing sample bounding boxes.

### Test Logger
```bash
cd src/output_module
python logger.py
```
Creates test logs and prints recent alerts.

### Test OutputModule
```bash
cd src/output_module
python output_handler.py
```
Runs full output processing test.

## Phase 1 Complete! ✅

You now have a working surveillance system:
- ✅ Input: Webcam or video file
- ✅ Detection: YOLOv8 finds people
- ✅ Tracking: DeepSORT assigns persistent IDs
- ✅ Behavior: Rule-based classification
- ✅ Output: Visualization + logging + dashboard

## Next Steps

**Phase 2** will add:
- Machine Learning behavior classification
- Anomaly detection
- Historical analysis
- Advanced heatmaps
