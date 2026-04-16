# Architecture Design Notes

## Overview

This document captures design decisions, trade-offs, and reasoning behind the Real-Time Intelligent Surveillance System architecture.

---

## Design Philosophy

### 1. Modularity Over Monolith

**Decision:** Design as five independent modules with clear interfaces.

**Rationale:**
- Enables testing individual components
- Allows swapping implementations (e.g., YOLOv8 → YOLOv10)
- Supports different deployment configurations
- Facilitates parallel development

**Trade-off:** Slight overhead from inter-module communication vs. tightly coupled code.

---

## Technology Selections

### Detection: YOLOv8

**Alternatives Considered:**
- YOLOv5: Older architecture, lower accuracy
- YOLOv11: Newer but less stable ecosystem
- Faster R-CNN: More accurate but 10x slower
- SSD MobileNet: Fast but lower accuracy on small objects

**Selection Reason:**
- Best speed/accuracy trade-off for real-time surveillance
- Native Python support via Ultralytics
- Active community and documentation
- Easy model size variants (n/s/m/l/x)

**Selected Model:** YOLOv8n (nano)
- **Speed:** ~100 FPS on GPU, ~20 FPS on CPU
- **Accuracy:** 37.3% COCO mAP (sufficient for person detection)
- **Size:** 6M parameters, 23MB weights

---

### Tracking: DeepSORT

**Alternatives Considered:**
- SORT: Simpler, no appearance features, fails during occlusion
- ByteTrack: Better for high-FPS, but appearance features help re-identification
- FairMOT: Joint detection and tracking, more complex
- BoT-SORT: Newer but less mature Python implementations

**Selection Reason:**
- Battle-tested in research and production
- Appearance features enable identity preservation through occlusion
- Clear separation between motion and appearance cues
- Available Python implementations

**Trade-off:** Requires separate feature extraction (computation cost) but improves tracking robustness.

---

### Behavior Analysis: Rule-Based First

**Decision:** Implement rule-based classification before ML-based.

**Rationale:**
| Aspect | Rule-Based | ML-Based |
|--------|-----------|----------|
| Development Time | 1-2 days | 1-2 weeks |
| Training Data | None required | Needs labeled dataset |
| Interpretability | High (human-readable rules) | Low (black box) |
| Accuracy Ceiling | Lower | Higher with good data |
| Edge Cases | Brittle | More robust |

**Hybrid Approach:**
- Start with rule-based for immediate functionality
- Collect data from rule-based system
- Train ML classifier using collected data
- A/B test and potentially switch or ensemble

---

## Pipeline Architecture Decisions

### Synchronous vs Asynchronous Processing

**Decision:** Hybrid approach
- Input: Async (threaded capture, never blocks)
- Processing: Sync (sequential pipeline, simpler state management)
- Output: Async (logging and alerting don't block main flow)

**Rationale:**
- Video processing is inherently sequential (temporal dependencies)
- Async at boundaries handles I/O variance
- Sync in core simplifies debugging

### Frame Skipping Strategy

**Decision:** Process every Nth frame (default: every frame for detection, every frame for tracking updates).

**Alternatives:**
- Detect every frame: Best accuracy, may lag on slow hardware
- Detect every 2nd/3rd frame: Faster, interpolate or use motion-only update for skipped frames

**Selected:** Start with every frame, add skipping if performance requires.

---

## Data Flow Decisions

### Immutable Frames

**Decision:** Copy frames at module boundaries, never share references.

**Rationale:**
- Prevents race conditions
- Allows modules to modify frames (annotation) without side effects
- Simplifies debugging (clear ownership)

**Trade-off:** Memory overhead (640×480×3 bytes ≈ 900KB per copy)

### Track History Management

**Decision:** Fixed-size deque (circular buffer) per track, default 30 frames.

**Rationale:**
- Bounded memory (no growth over time)
- FIFO semantics match temporal analysis needs
- 1 second at 30 FPS sufficient for behavior classification

---

## Real-Time Performance Strategy

### Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| End-to-end latency | < 200ms | > 500ms (unacceptable) |
| Processing FPS | > 15 FPS | < 5 FPS (fails real-time) |
| Memory usage | < 2GB | > 4GB (risk on edge devices) |

### Optimization Techniques (Priority Order)

1. **Resolution Scaling**: Run detection at 640×480 or lower
2. **Model Selection**: Use YOLOv8n (fastest variant)
3. **Frame Skipping**: Process every 2nd frame if needed
4. **GPU Acceleration**: CUDA for PyTorch models
5. **Half Precision**: FP16 inference (2x speedup on modern GPUs)
6. **TensorRT**: Export to optimized inference engine for deployment

### Queue Management

**Decision:** Bounded queues with drop policy.

| Queue | Size | Overflow Policy | Reason |
|-------|------|-----------------|--------|
| Frame buffer | 5 | Drop oldest | Prevent memory growth |
| Alert queue | 10 | Drop newest | Prevent alert spam |

---

## Extensibility Design

### Plugin Architecture

Each module defines abstract base class:

```python
class BaseDetectionModule(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

# YOLO implementation
class YOLODetectionModule(BaseDetectionModule):
    ...

# Future: Custom detector
class CustomDetectionModule(BaseDetectionModule):
    ...
```

This enables:
- Swapping YOLO for custom detector
- Multiple behavior analysis strategies
- Different output channels (database, cloud, etc.)

### Configuration-Driven

All parameters externalized to YAML:
- Enables tuning without code changes
- Supports multiple deployment profiles (dev/staging/prod)
- Documentation embedded in config schema

---

## Error Handling Philosophy

### Fail-Fast vs. Fail-Graceful

| Component | Strategy | Reason |
|-----------|----------|--------|
| Input | Fail-Graceful | Camera disconnect is recoverable |
| Detection | Fail-Fast | Model errors indicate configuration issues |
| Tracking | Fail-Graceful | Can continue with degraded tracking |
| Behavior | Fail-Silent | Missing behavior analysis shouldn't crash system |
| Output | Fail-Graceful | Logging failure shouldn't stop surveillance |

### Recovery Mechanisms

1. **Camera Disconnect**: Retry with exponential backoff, alert operator
2. **Model Inference Error**: Skip frame, log error, continue
3. **Memory Pressure**: Reduce track buffer size, increase frame skipping
4. **Queue Overflow**: Log warning, drop data per policy

---

## Research vs Production Balance

### Research Components (Advanced Features)

1. **ML-Based Behavior Classifier**
   - Enables publication-worthy results
   - Requires dataset and training infrastructure
   - Adds complexity but improves accuracy

2. **Activity Heatmaps**
   - Visual research output showing spatiotemporal patterns
   - Useful for security planning papers
   - Demonstrates system sophistication

### Production Components (Core System)

1. **Rule-Based Behavior Detection**
   - Fast, reliable, interpretable
   - Sufficient for most surveillance needs

2. **JSON Logging**
   - Industry standard for event storage
   - Enables audit trails and analytics

3. **Streamlit Dashboard**
   - Lightweight web UI
   - Easier deployment than custom frontend

---

## Security Considerations

### Data Privacy

**Decision:** Process frames in memory, don't store raw video unless configured.

**Rationale:**
- Surveillance video contains PII
- Alert clips (short duration) are reasonable
- Full video storage requires legal/privacy review

### Configuration

```yaml
privacy:
  store_raw_video: false
  store_alert_clips: true
  clip_duration_seconds: 5
  blur_faces: false  # Could add face blurring
```

---

## Testing Strategy

### Module Unit Tests

Each module tested in isolation:
- Input: Mock video file
- Detection: Static test images
- Tracking: Synthetic frame sequences
- Behavior: Controlled tracklet patterns
- Output: Mock output collectors

### Integration Tests

Full pipeline tests:
- Short video file (10 seconds)
- Verify end-to-end latency
- Check track ID consistency
- Validate alert generation

### Performance Benchmarks

Continuous benchmarks:
- FPS measurement
- Memory profiling
- Latency histograms

---

## Future Evolution

### Phase 1: Current (Core System)
- Detection + Tracking + Rule-based behavior
- Streamlit dashboard
- JSON logging

### Phase 2: Enhanced Analytics
- ML behavior classifier
- Activity heatmaps
- Multi-camera support

### Phase 3: Production Hardening
- REST API for integration
- Docker containerization
- Cloud deployment (AWS/GCP)

### Phase 4: Advanced Research
- Skeleton-based action recognition
- Multi-person interaction analysis
- Anomaly detection via deep learning

---

## Open Questions

1. **Dataset for ML training**: Collect custom or use public (NTU RGB+D)?
2. **Camera calibration**: Fix for perspective distortion in behavior analysis?
3. **Night/low-light**: Add IR camera support or low-light enhancement?
4. **Multi-camera tracking**: Re-identification across non-overlapping cameras?

---

## References

1. YOLOv8: Jocher et al., 2023 (Ultralytics)
2. DeepSORT: Wojke et al., 2017 (ICCW)
3. Kalman Filter: Welch & Bishop, 1995
4. Behavior Analysis: Survey on Human Action Recognition (Kong & Fu, 2018)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Current | Initial architecture notes |
