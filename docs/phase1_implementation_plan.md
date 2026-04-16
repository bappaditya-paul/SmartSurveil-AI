# Phase 1 Implementation Plan: Core System + Dashboard

**Objective:** Build a working real-time surveillance system with human detection, tracking, and basic dashboard for research internship demonstration.

**Timeline:** 3-4 weeks (adjust based on your schedule)

**Deliverable:** Functional system that can detect, track, and display humans in real-time with behavior annotations

---

## Week 1: Foundation & Detection Pipeline

### Day 1-2: Project Setup & Input Module
```
Tasks:
□ Finalize directory structure
□ Create config loader utility
□ Implement InputModule (camera/video file capture)
□ Add frame preprocessing (resize, normalize)
□ Test: Display webcam feed with FPS counter

Key Files:
- src/utils/config_loader.py
- src/input_module/input_handler.py
- tests/test_input_module.py

Checkpoint: Can run "python -m src.input_module" and see live video
```

### Day 3-4: Detection Module (YOLOv8)
```
Tasks:
□ Integrate YOLOv8 from ultralytics
□ Create DetectionModule class
□ Implement detection post-processing
□ Add bbox scaling (model resolution → frame resolution)
□ Test on sample images and video

Key Files:
- src/detection_module/detector.py
- src/detection_module/detection_result.py
- tests/test_detection.py

Checkpoint: Run detection on video, save annotated output, verify accuracy
```

### Day 5-7: Integration Week 1 Components
```
Tasks:
□ Connect Input → Detection pipeline
□ Create main.py skeleton
□ Add visualization overlay for detections
□ Implement FPS benchmarking
□ Debug and optimize

Key Files:
- app/main.py (basic version)
- src/utils/visualization.py (bbox drawing)

Checkpoint: Run "python app/main.py" and see real-time detection with bounding boxes
```

---

## Week 2: Tracking & Behavior Analysis

### Day 8-10: Tracking Module (DeepSORT)
```
Tasks:
□ Install deep_sort_realtime package
□ Create TrackingModule wrapper
□ Implement track state management
□ Add track history buffer
□ Integrate with Detection module

Key Files:
- src/tracking_module/tracker.py
- src/tracking_module/track.py
- tests/test_tracking.py

Key Concepts:
- Track ID persistence across frames
- Handle track birth/death lifecycle
- Store last N positions for each track

Checkpoint: Multiple people tracked with consistent IDs, trajectory trails visible
```

### Day 11-12: Behavior Analysis (Rule-Based)
```
Tasks:
□ Implement motion feature extraction (velocity, aspect ratio)
□ Create RuleBasedClassifier
□ Define behavior rules (walking, running, falling)
□ Add alert conditions
□ Connect to tracking data

Key Files:
- src/behavior_module/behavior_analyzer.py
- src/behavior_module/rule_engine.py
- src/behavior_module/behavior_result.py

Behavior Rules to Implement:
- Walking: velocity 2-8 px/frame
- Running: velocity > 8 px/frame
- Falling: aspect ratio < 1.0 (sudden change from >2.0)

Checkpoint: System classifies behaviors and displays labels on screen
```

### Day 13-14: Integration & Testing
```
Tasks:
□ Full pipeline: Input → Detection → Tracking → Behavior
□ Test with recorded video samples
□ Test with live webcam
□ Fix integration bugs
□ Document any issues for research notes

Key Files:
- app/main.py (integrated version)
- tests/test_full_pipeline.py

Checkpoint: Complete system running end-to-end with real-time performance
```

---

## Week 3: Output Module & Dashboard

### Day 15-17: Visualizer & Logger
```
Tasks:
□ Create Visualizer class (bbox, labels, trajectories)
□ Implement EventLogger (JSON line format)
□ Add alert generation logic
□ Create simple heatmap accumulator
□ Test visualization quality

Key Files:
- src/output_module/visualizer.py
- src/output_module/logger.py
- src/output_module/heatmap_generator.py

Log Format:
{
    "timestamp": 1234567890.123,
    "track_id": 5,
    "behavior": "falling",
    "confidence": 0.89,
    "position": [320, 240]
}

Checkpoint: Logs being written to data/logs/, visual overlays look professional
```

### Day 18-19: Streamlit Dashboard (Basic)
```
Tasks:
□ Create app/dashboard.py
□ Implement live video display
□ Add metrics sidebar (FPS, active tracks, total detections)
□ Display recent alerts table
□ Add system controls (start/stop, config display)

Dashboard Features:
- Main area: Live video feed with detections
- Sidebar: System metrics and controls
- Alert panel: Recent events with timestamps

Checkpoint: Can launch dashboard with "streamlit run app/dashboard.py"
```

### Day 20-21: Dashboard Enhancement
```
Tasks:
□ Add heatmap visualization tab
□ Implement historical data viewer (past alerts)
□ Add configuration display/editing
□ Test dashboard performance
□ Mobile responsiveness check

Key Files:
- app/dashboard.py (complete)
- src/output_module/dashboard_adapter.py

Checkpoint: Professional-looking dashboard, all features functional
```

---

## Week 4: Testing, Documentation & Polish

### Day 22-24: Testing & Optimization
```
Tasks:
□ Write unit tests for each module
□ Create integration test suite
□ Performance profiling (identify bottlenecks)
□ Optimize if FPS < 15 (adjust resolution, model size)
□ Test edge cases (empty frames, occlusions, lighting changes)

Tests to Write:
- test_input_module.py: Camera/file loading
- test_detection.py: YOLO accuracy, NMS
- test_tracking.py: ID consistency, track lifecycle
- test_behavior.py: Rule classification accuracy
- test_integration.py: Full pipeline

Checkpoint: pytest passes all tests, system stable for 10+ minutes
```

### Day 25-26: Documentation
```
Tasks:
□ Write module docstrings
□ Create usage examples
□ Update README with Phase 1 features
□ Document known limitations
□ Create demo script

Documentation Files:
- README.md (updated)
- docs/usage_guide.md
- examples/demo_video.py
- examples/demo_webcam.py

Checkpoint: Another person can run the system following README
```

### Day 27-28: Demo Preparation
```
Tasks:
□ Record demo video of system working
□ Prepare sample test videos
□ Create presentation slides (if needed)
□ Practice demo walkthrough
□ Backup working version (git tag v1.0-phase1)

Deliverables:
- Working system with webcam
- Demo with test video file
- 5-minute presentation outline
- Research notes document

Checkpoint: Ready to demonstrate to advisor/mentor
```

---

## File Creation Order (Priority)

### Week 1 Files (Create in this order):
1. `src/utils/config_loader.py` - Load YAML config
2. `src/input_module/input_handler.py` - Video capture
3. `src/detection_module/detector.py` - YOLO wrapper
4. `app/main.py` - Basic pipeline test

### Week 2 Files:
5. `src/tracking_module/tracker.py` - DeepSORT wrapper
6. `src/behavior_module/rule_engine.py` - Simple rules
7. `src/behavior_module/behavior_analyzer.py` - Integration
8. Update `app/main.py` - Full pipeline

### Week 3 Files:
9. `src/output_module/visualizer.py` - Drawing functions
10. `src/output_module/logger.py` - JSON logging
11. `app/dashboard.py` - Streamlit UI

### Week 4 Files:
12. `tests/` - Test files
13. Documentation updates

---

## Research Internship Deliverables

### Minimum Viable Product (MVP):
- [ ] Real-time person detection (YOLOv8)
- [ ] Multi-person tracking (DeepSORT)
- [ ] Basic behavior classification (3 behaviors)
- [ ] Live dashboard with metrics
- [ ] JSON event logging

### Research Components:
- [ ] Performance benchmark report (FPS, latency)
- [ ] Behavior classification accuracy analysis
- [ ] System limitations documented
- [ ] Future improvements roadmap

### Presentation Materials:
- [ ] Live demo capability
- [ ] Architecture diagram (from docs/)
- [ ] Results slide (metrics, screenshots)
- [ ] Challenges & learnings

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| YOLO too slow | Use YOLOv8n (nano), reduce resolution to 416x320 |
| Tracking fails | Tune DeepSORT params, reduce max_age |
| Low accuracy | Increase confidence threshold, add NMS tuning |
| Camera issues | Prepare test video files as fallback |
| Dashboard lag | Reduce refresh rate, use st.empty() efficiently |

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Detection FPS | >15 | Display FPS counter |
| Tracking MOTA | >70% | Manual check on 1-min video |
| Behavior Accuracy | >80% | Test with known actions |
| System Uptime | >30 min | Run continuously |
| Alert Latency | <2 sec | Stopwatch test |

---

## Phase 1 Complete Definition

Phase 1 is complete when:
1. ✅ System runs for 30 minutes without crash
2. ✅ Can detect & track 3+ people simultaneously
3. ✅ Classifies walking/running/falling correctly
4. ✅ Dashboard displays live feed + metrics
5. ✅ Logs events to JSON files
6. ✅ All tests pass
7. ✅ Demo ready for presentation

---

## Phase 2 Preview (Future Work)

Once Phase 1 is stable:
- ML-based behavior classifier (Random Forest/LSTM)
- Activity heatmaps
- Multi-camera support
- REST API for integration
- Cloud deployment

---

## Daily Check-in Questions

Each day, ask yourself:
1. What did I complete today?
2. What's blocking me?
3. What do I need for tomorrow?
4. Should I adjust the timeline?

---

**Start Date:** ___/___/___
**Target Completion:** ___/___/___
**Advisor Check-ins:** Weekly recommended

Good luck with your research internship!
