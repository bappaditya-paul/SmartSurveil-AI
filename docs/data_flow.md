# Data Flow Documentation

## Overview

This document describes how data moves through the Real-Time Intelligent Surveillance System, including data formats, transformation stages, and inter-module communication.

---

## Data Flow Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                                  │
└────────────────────────────────────────────────────────────────────────────┘

Camera Frame (BGR Image)
         │
         ▼
┌─────────────────┐
│  INPUT MODULE   │
│                 │
│ • Capture       │
│ • Resize        │
│ • Normalize     │
└────────┬────────┘
         │ Frame: numpy.ndarray[H×W×3]
         │ Metadata: timestamp, source_id, fps
         ▼
┌─────────────────┐
│ DETECTION MODULE│
│                 │
│ • YOLOv8 Inf.   │
│ • NMS Filter    │
│ • Class Filter  │
└────────┬────────┘
         │ Detections: List[(bbox, conf, class)]
         │ bbox: [x1, y1, x2, y2]
         ▼
┌─────────────────┐
│ TRACKING MODULE │
│                 │
│ • DeepSORT      │
│ • Kalman Filter │
│ • Feature Ext.  │
└────────┬────────┘
         │ Tracks: List[(track_id, bbox, history, age)]
         │ history: deque of last N positions
         ▼
┌─────────────────┐
│ BEHAVIOR MODULE │
│                 │
│ • Feature Ext.  │
│ • Classification│
│ • Alert Logic   │
└────────┬────────┘
         │ BehaviorEvents: List[(track_id, type, conf, alert)]
         ▼
┌─────────────────┐
│ OUTPUT MODULE   │
│                 │
│ • Render        │
│ • Log           │
│ • Alert         │
│ • Heatmap       │
└─────────────────┘
```

---

## Stage-by-Stage Data Transformation

### Stage 1: Input Acquisition

**Input:** Raw video stream (camera, file, or RTSP)

**Processing:**
1. Frame capture via OpenCV
2. Resize to processing resolution (default: 640×480)
3. Convert to RGB if needed
4. Add timestamp and frame metadata

**Output Data Structure:**
```python
{
    "frame": numpy.ndarray,  # shape: (H, W, 3), dtype: uint8
    "timestamp": float,       # epoch time
    "source_id": str,         # camera identifier
    "metadata": {
        "original_resolution": (int, int),
        "processing_resolution": (int, int),
        "fps": float
    }
}
```

---

### Stage 2: Object Detection

**Input:** Preprocessed frame

**Processing:**
1. YOLOv8 inference
2. Non-Maximum Suppression (NMS)
3. Class filtering (person only)

**Output Data Structure:**
```python
[
    {
        "bbox": [x1, y1, x2, y2],  # pixel coordinates
        "confidence": float,        # 0.0 - 1.0
        "class_id": int,            # 0 for person
        "class_name": str           # "person"
    }
]
```

**Transformation Notes:**
- Raw YOLO outputs are scaled back to original frame dimensions
- Confidence threshold filters low-quality detections (default: 0.5)
- NMS removes overlapping boxes (IoU threshold: 0.45)

---

### Stage 3: Multi-Object Tracking

**Input:** List of detections + previous track states

**Processing:**
1. Kalman filter prediction for existing tracks
2. Hungarian algorithm matching (detections ↔ tracks)
3. Update matched tracks with new observations
4. Create new tracks for unmatched detections
5. Mark old tracks as deleted

**Output Data Structure:**
```python
[
    {
        "track_id": int,              # persistent ID
        "bbox": [x1, y1, x2, y2],
        "confidence": float,
        "state": str,                 # "confirmed" | "tentative" | "deleted"
        "age": int,                   # frames since creation
        "history": deque,             # last 30 centroid positions
        "velocity": [vx, vy],         # pixels per frame
        "feature_vector": ndarray     # appearance embedding
    }
]
```

**Track State Machine:**
- **Tentative**: New track, needs 3 hits to confirm
- **Confirmed**: Active track with consistent detections
- **Deleted**: Track lost for >30 frames, removed

---

### Stage 4: Behavior Analysis

**Input:** Active tracks with position history

**Processing:**
1. Extract motion features (velocity, acceleration, aspect ratio)
2. Compute trajectory descriptors
3. Apply rule-based classification OR ML inference
4. Evaluate alert conditions

**Output Data Structure:**
```python
[
    {
        "track_id": int,
        "behavior": str,              # "walking" | "running" | "falling" | ...
        "confidence": float,
        "features": {
            "velocity_magnitude": float,
            "acceleration": float,
            "aspect_ratio": float,
            "direction_variance": float
        },
        "alert_triggered": bool,
        "alert_type": str | None      # "fall_detected" | "suspicious_activity"
    }
]
```

---

### Stage 5: Output Generation

**Input:** Annotated frame, behavior events, system metrics

**Processing:**
1. Render visual overlays (bboxes, IDs, labels)
2. Update dashboard (Streamlit)
3. Log events to file
4. Trigger alerts if conditions met
5. Update heatmap accumulators

**Output Channels:**

| Channel | Data Format | Destination |
|---------|-------------|-------------|
| Display | Rendered frame (ndarray) | Screen / Streamlit |
| Video | MP4/H264 | File storage |
| Logs | JSON Lines | `logs/events_YYYY-MM-DD.jsonl` |
| Alerts | JSON payload | Dashboard, optional webhook |
| Heatmap | PNG overlay | Dashboard, file export |

**Log Entry Structure:**
```json
{
    "timestamp": 1713024000.123,
    "event_type": "behavior_alert",
    "track_id": 5,
    "behavior": "falling",
    "confidence": 0.89,
    "position": [320, 240],
    "alert_type": "fall_detected"
}
```

---

## Inter-Module Communication

### Synchronous vs Asynchronous

| Flow | Pattern | Queue Size | Overflow Policy |
|------|---------|------------|-----------------|
| Input → Detection | Async (threaded) | 5 frames | Drop oldest |
| Detection → Tracking | Sync | N/A | Block if needed |
| Tracking → Behavior | Sync | N/A | Block if needed |
| All → Output | Async | 10 events | Drop newest |

### Thread Safety

- **Shared State**: Track database (protected by lock)
- **Immutable Data**: Frames are copied at module boundaries
- **Queue Management**: Thread-safe queues for producer-consumer pairs

---

## Data Retention and Buffer Management

| Data Type | Buffer Size | Eviction Policy |
|-----------|-------------|-----------------|
| Frame history | 1 frame | Overwrite |
| Track positions | 30 frames (1 sec @ 30fps) | FIFO |
| Behavior events | 100 events | FIFO |
| Heatmap grid | 300 seconds exponential decay | Time-weighted |
| Alert history | 24 hours | Time-based cleanup |

---

## Error Handling Flow

```
Module Error Detected
       │
       ▼
┌──────────────┐
│ Log Error    │
│ (severity)   │
└──────┬───────┘
       │
       ▼
   Recoverable?
    /        \
   Yes       No
   │          │
   ▼          ▼
┌──────┐  ┌────────┐
│Skip  │  │Graceful│
│Frame │  │Shutdown│
└──────┘  └────────┘
```

---

## Performance Metrics Flow

System continuously collects:
- **FPS**: Frames processed per second
- **Latency**: Time from frame capture to output (target < 200ms)
- **Detection rate**: Detections per frame
- **Track count**: Active tracks
- **Behavior distribution**: Histogram of detected behaviors

Metrics are exposed to dashboard via shared metrics object updated every frame.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Current | Initial data flow specification |
